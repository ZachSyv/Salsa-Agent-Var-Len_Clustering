from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch
from models.training_utils import *
import numpy as np
import models.vqvae as vqvae
import re

PAIR = True

for_old_model_inference = True
class MotionLLM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm_backbone)
        self.llm = AutoModelForCausalLM.from_pretrained(self.args.llm_backbone)
        self.nb_text_tokens = len(self.tokenizer)

        self.device = args.device

        self.lora_config_t2m = LoraConfig(
            r=self.args.lora_r_t2m,
            lora_alpha=self.args.lora_alpha_t2m,
            target_modules=[ #'embed_tokens', 'lm_head',
                            'o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # trainable_token_indices=[20]
        )
        if args.is_baseline:
            self.lora_config_m2t = LoraConfig(
                r=self.args.lora_r_m2t,
                lora_alpha=self.args.lora_alpha_m2t,
                target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
                lora_dropout=self.args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                # trainable_token_indices=[257000] # PEFT version 15 support this but not guaranteed.
            )

        self.load_motionvq()  # HumanML3D VQVAE; used for humanml3d path and baseline caption/generate
        self.motion_repr_type = getattr(args, 'motion_repr_type', 'humanml3d')  # 'humanml3d' | 'interhuman'
        self.include_audio = getattr(args, 'include_audio', False)  # if True, add <Audio_0>.. tokens

        # -------------------------------------------------------------------------
        # Baseline: HumanML3D-style tokens only (Motion, </Motion>, <Motion_i>)
        # -------------------------------------------------------------------------
        if args.is_baseline:
            self.tokenizer.add_tokens(['<Motion>', '</Motion>'])
            self.motion_token_indices = len(self.tokenizer) + np.arange(self.args.nb_code)
            for i in range(self.args.nb_code):
                self.tokenizer.add_tokens([f'<Motion_{i}>'])
            self.llm.resize_token_embeddings(len(self.tokenizer))
            self.llm = get_peft_model(self.llm, self.lora_config_t2m, adapter_name='t2m')
            self.llm.add_adapter('m2t', self.lora_config_m2t)

        # -------------------------------------------------------------------------
        # HumanML3D (Salsa): previous implementation — Motion + Salsa modalities
        # -------------------------------------------------------------------------
        elif self.motion_repr_type == 'humanml3d':
            self.tokenizer.add_tokens(['<Motion>', '</Motion>'])
            self.motion_token_indices = len(self.tokenizer) + np.arange(self.args.nb_code)
            for i in range(self.args.nb_code):
                self.tokenizer.add_tokens([f'<Motion_{i}>'])
            humanml3d_special_tokens = [
                "<LeaderScript>", "</LeaderScript>",
                "<FollowerScript>", "</FollowerScript>",
                "<LeaderMotion>", "</LeaderMotion>",
                "<FollowerMotion>", "</FollowerMotion>",
            ]
            if self.include_audio:
                humanml3d_special_tokens += ["<AudioTokens>", "</AudioTokens>"] + [f"<Audio_{i}>" for i in range(4096)]
            self.tokenizer.add_tokens(humanml3d_special_tokens)
            self.llm.resize_token_embeddings(len(self.tokenizer))
            self.llm = get_peft_model(self.llm, self.lora_config_t2m, adapter_name='t2m')
            pefti_llm = get_peft_model(self.llm, self.lora_config_t2m, adapter_name='t2m')
            for name, param in pefti_llm.named_parameters():
                if 'lora' in name:
                    print(name)
            embeddings = self.llm.get_input_embeddings().weight[self.nb_text_tokens:]
            lm_head = self.llm.lm_head.weight[self.nb_text_tokens:]

        # -------------------------------------------------------------------------
        # InterHuman: use INTERHUMAN_SPECIAL_TOKENS from training_utils + IH/Rel + optional Audio
        # -------------------------------------------------------------------------
        elif self.motion_repr_type == 'interhuman':
            from models.training_utils import INTERHUMAN_SPECIAL_TOKENS
            interhuman_tokens = list(INTERHUMAN_SPECIAL_TOKENS)
            nb_ih = getattr(self.args, 'nb_ih_code', 512)
            nb_rel = getattr(self.args, 'nb_rel_code', 512)
            interhuman_tokens += [f"<IH_{i}>" for i in range(nb_ih)] + [f"<Rel_{i}>" for i in range(nb_rel)]
            if self.include_audio:
                interhuman_tokens += ["<AudioTokens>", "</AudioTokens>"]
                interhuman_tokens += [f"<Audio_{i}>" for i in range(4096)]
            self.tokenizer.add_tokens(interhuman_tokens)
            self.llm.resize_token_embeddings(len(self.tokenizer))
            self.motion_token_indices = None  # InterHuman uses IH/Rel; no Motion_i range
            self.llm = get_peft_model(self.llm, self.lora_config_t2m, adapter_name='t2m')
            pefti_llm = get_peft_model(self.llm, self.lora_config_t2m, adapter_name='t2m')
            for name, param in pefti_llm.named_parameters():
                if 'lora' in name:
                    print(name)
            embeddings = self.llm.get_input_embeddings().weight[self.nb_text_tokens:]
            lm_head = self.llm.lm_head.weight[self.nb_text_tokens:]

        else:
            raise ValueError(f"motion_repr_type must be 'humanml3d' or 'interhuman', got {self.motion_repr_type!r}")

        self.llm.to(self.device)
        self.llm.eval()

        # print(self.llm)

    def load_motionvq(self):
        self.mean = np.load('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')
        self.std = np.load('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')

        self.args.nb_joints = 22
        self.args.dataname = 't2m'
        self.args.vq_path = "ckpt/vqvae.pth"
        self.net = vqvae.HumanVQVAE(self.args, ## use args to define different parameters in different quantizers
                           self.args.nb_code,
                           self.args.code_dim,
                           self.args.output_emb_width,
                           self.args.down_t,
                           self.args.stride_t,
                           self.args.width,
                           self.args.depth,
                           self.args.dilation_growth_rate,
                           self.args.vq_act,
                           self.args.vq_norm)
        print ('loading vqvae from {}'.format(self.args.vq_path))
        ckpt = torch.load(self.args.vq_path, map_location='cpu')
        self.net.load_state_dict(ckpt['net'], strict=True)
        self.net.eval()
        self.net.to(self.device)
    
    def forward(self, level, ms_desc_L, ms_des_F, vq_tokens_L, vq_tokens_F, audio_tokens, batch_interhuman_data=None):

        # inputs_ids, targets, attention_mask = process_batch(tokenizer=self.tokenizer,
        #                                                     batch_of_captions=caption,
        #                                                     max_tgt_len=900,
        #                                                     batch_of_motions=motion_tokens,
        #                                                     batch_of_motionscript=ms_desc_bins,
        #                                                     batch_of_audio=audio_tokens)
        inputs_ids, targets, attention_mask = process_batch_Salsa(
            tokenizer=self.tokenizer,
            batch_aux_info=level,
            batch_ms_desc_L=ms_desc_L,
            batch_ms_des_F=ms_des_F,
            batch_vq_tokens_L=vq_tokens_L,
            batch_vq_tokens_F=vq_tokens_F,
            batch_audio_tokens=audio_tokens,
            max_tgt_len=700,
            current_batch_task=None if (getattr(self.args, 'task', None) in (None, 'none', 'all')) else self.args.task,
            motion_repr_type=self.motion_repr_type,
            batch_interhuman_data=batch_interhuman_data,
            include_audio=getattr(self.args, 'include_audio', False),
        )



        # print(inputs_ids.shape)
        # print(targets.shape)
        # print(attention_mask.shape)
        # print(tokenizer.decode(inputs_ids[0]))
        inputs_ids = inputs_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        targets = targets.to(self.device)

        outputs = self.llm(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        loss = outputs.loss
        # print(outputs.logits.shape)
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return loss, gen_acc, chosen_tokens, labels
    
    def generate(self, caption):
        self.llm.set_adapter('t2m')
        self.llm.eval()
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
        input_text = '### Input:\n' + caption + '\n\nResponse: <Motion>'
        input = prompt + instruction + input_text
        input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        outputs = self.llm.generate(
            input_ids, 
            max_length=200, 
            num_beams=2, 
            early_stopping=True, 
            return_dict_in_generate=True, 
            output_scores=True
        )

        scores = torch.stack(outputs.scores)  # [num_generated_tokens, num_beams, vocab_size]
        # print(scores.shape)
        # Take only the best beam (beam 0)
        best_beam_scores = scores[:, 0, :]  # [num_generated_tokens, vocab_size]
        motion_logits = best_beam_scores[:, -(self.args.nb_code+2):]
        # print(motion_logits.shape)
        motion_tokens = torch.argmax(motion_logits, dim=-1)  # [num_generated_tokens]
        # print(motion_tokens)
        # Remove end_of_motion token (index=1) if present
        if 1 in motion_tokens:
            motion_tokens = motion_tokens[:motion_tokens.tolist().index(1)]
        # Ensure tokens don't go below 0 when adjusting for special tokens
        motion_tokens = torch.clamp(motion_tokens - 2, min=0)  # remove the first two special tokens while preventing negative values
        
        # print(motion_tokens)
        return motion_tokens

    def generate_Payam(self, full_prompt, inputs_ids):
        self.llm.set_adapter('t2m')
        self.llm.eval()
        # inputs_ids, targets, attention_mask = process_batch_Salsa(tokenizer=self.tokenizer,
        #                                                           batch_aux_info=level,
        #                                                           batch_ms_desc_L=ms_desc_L,
        #                                                           batch_ms_des_F=ms_des_F,
        #                                                           batch_vq_tokens_L=vq_tokens_L,
        #                                                           batch_vq_tokens_F=vq_tokens_F,
        #                                                           batch_audio_tokens=audio_tokens,
        #                                                           max_tgt_len=700,
        #                                                           current_batch_task=self.args.task,
        #                                                           )

        # inputs_ids = inputs_ids.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        # targets = targets.to(self.device)
        # Todo: --------------------------------------------------------------------------------------------------------
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        outputs = self.llm.generate(
            input_ids,
            max_length=200, # todo: 200 for baseline 700 for ours
            num_beams=2,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
        )

        scores = torch.stack(outputs.scores)  # [num_generated_tokens, num_beams, vocab_size]
        # print(scores.shape)
        # Take only the best beam (beam 0)
        best_beam_scores = scores[:, 0, :]  # [num_generated_tokens, vocab_size]
        motion_logits = best_beam_scores[:, -(self.args.nb_code + 2):]
        # print(motion_logits.shape)
        motion_tokens = torch.argmax(motion_logits, dim=-1)  # [num_generated_tokens]
        # print(motion_tokens)
        # Remove end_of_motion token (index=1) if present
        if 1 in motion_tokens:
            motion_tokens = motion_tokens[:motion_tokens.tolist().index(1)]
        # Ensure tokens don't go below 0 when adjusting for special tokens
        motion_tokens = torch.clamp(motion_tokens - 2,
                                    min=0)  # remove the first two special tokens while preventing negative values
        if for_old_model_inference: return motion_tokens

        # todo: this is necessary to make sure motion tokens are parsed correctly since we added more tokens e.g., audio
        pred_logits = best_beam_scores
        pred_tokens = torch.argmax(pred_logits, dim=-1)
        pred_txt = self.tokenizer.decode(pred_tokens)
        indices = [int(x) for x in re.findall(r'<Motion_(\d+)', pred_txt)]

        tensor_indices = torch.tensor(indices, device='cuda:0')

        # print(motion_tokens)
        return tensor_indices # motion_tokens

    def generate_Payam_interhuman(self, full_prompt, task, max_new_tokens=150, num_beams=2, do_sample=False):
        """Generate InterHuman/Relationship token sequence from prompt (InterHuman representation only).
        full_prompt: prompt text ending with '### Response:\\n' + label + open delimiter + space (e.g. 'Follower motion: <FollowerMotion> ').
        task: one of INTERHUMAN_TASKS (e.g. 'leader_rel_to_follower').
        Returns dict with keys leader_tokens, follower_tokens, relationship_tokens; only the predicted output type is filled (list of ints), others None.
        """
        from models.training_utils import INTERHUMAN_TASK_OUTPUT_TYPE, INTERHUMAN_TASKS
        if self.motion_repr_type != 'interhuman':
            raise ValueError("generate_Payam_interhuman requires motion_repr_type='interhuman'")
        out_type = INTERHUMAN_TASK_OUTPUT_TYPE.get(task, "follower")
        self.llm.set_adapter('t2m')
        self.llm.eval()
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        input_len = input_ids.shape[1]
        with torch.no_grad():
            outputs = self.llm.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_ids = outputs.sequences[0][input_len:]
        pred_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        ih_matches = re.findall(r'<IH_(\d+)>', pred_text)
        rel_matches = re.findall(r'<Rel_(\d+)>', pred_text)
        ih_tokens = [int(x) for x in ih_matches]
        rel_tokens = [int(x) for x in rel_matches]
        result = {"leader_tokens": None, "follower_tokens": None, "relationship_tokens": None}
        if out_type == "relationship":
            result["relationship_tokens"] = rel_tokens
        elif out_type in ("leader", "follower"):
            result[f"{out_type}_tokens"] = ih_tokens
        return result

    def caption(self, motion):
        self.llm.set_adapter('m2t')
        self.llm.eval()
        motion = self.normalize(motion)
        motion = torch.from_numpy(motion).float().to(self.device).unsqueeze(0)
        motion_tokens = self.net.encode(motion).squeeze(0)
        motion_tokens = motion_tokens + self.nb_text_tokens + 2 # reindex the motion tokens
        # print(motion_tokens)

        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction = "### Instruction:\nGenerate a caption matching the following input human motion token sequence.\n\n"
        input_text = '### Input:\n' + "<Motion>" + self.tokenizer.decode(motion_tokens) + '</Motion>' + '\n\nResponse: '
        input_texts = prompt + instruction + input_text
        # print(input_texts)
        input_ids = self.tokenizer.encode(input_texts, return_tensors="pt").to(self.device)
        pred = self.llm.generate(
            input_ids, 
            max_length=200, 
            num_beams=2
        )
        pred = pred[0, len(input_ids[0]):]
        pred = self.tokenizer.decode(pred)
        caption = pred.split('<eos>')[0]

        return caption
    
    def save_model(self, path):
        # only save the lora weights of the model
        save_dict = {}
        for name, param in self.llm.named_parameters():
            if 'lora' in name:
                save_dict[name] = param

        # save the additional token embeddings
        embeddings = self.llm.get_input_embeddings().weight[self.nb_text_tokens:]
        save_dict['embeddings'] = embeddings

        # save the lm_head of the additional tokens
        lm_head = self.llm.lm_head.weight[self.nb_text_tokens:]
        save_dict['lm_head'] = lm_head

        torch.save(save_dict, path)

    def load_model(self, path):
        print(f"Loading model from {path}")
        save_dict = torch.load(path, map_location=self.device)
        for name, param in self.llm.named_parameters():
            # print(name)
            if name in save_dict:
                param.data = save_dict[name]
        self.llm.get_input_embeddings().weight.data[self.nb_text_tokens:] = save_dict['embeddings']
        self.llm.lm_head.weight.data[self.nb_text_tokens:] = save_dict['lm_head']

    def denormalize(self, motion):
        return self.mean + motion * self.std

    def normalize(self, motion):
        return (motion - self.mean) / self.std
    
