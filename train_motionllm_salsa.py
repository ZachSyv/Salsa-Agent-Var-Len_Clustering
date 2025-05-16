import os
import torch
import wandb
from tqdm import tqdm
from models.mllm import MotionLLM
from options.option_llm import get_args_parser
from torch.utils.data import DataLoader
from utils.salsa_utils.salsa_dataloader import Salsa_Dataset
def print_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ---- %{(100*(trainable_params/total_params)):.2f}")

PAIR2LEVEL = {
    f"pair{i}": level
    for i, level in zip(range(1, 10), ["beginner", "intermediate", "professional"] * 3)
}

def train(model, train_loader, args):
    model.train()

    model.llm.set_adapter('t2m')  # activate text-to-motion adapter
    #Todo Clone 't2m' adapter weights into 't2m-salsa' for continued fine-tuning
    # model.llm.add_adapter('t2m-salsa', model.lora_config_t2m)
    # model.llm.adapters['t2m-salsa'].load_state_dict(
    #     model.llm.adapters['t2m'].state_dict())

    # Switch to the new adapter
    # model.llm.set_adapter('t2m-salsa')
    model.llm.print_trainable_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_acc = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            # caption, motion = batch[0], batch[1]  # assumes your custom DataLoader outputs this

            # caption, ms_desc_bins, audio_tokens, motion_tokens = batch

            level, ms_desc_L, ms_des_F, vq_tokens_L, vq_tokens_F, audio_tokens, aux_batch = batch
            # level = PAIR2LEVEL[(aux_info['vid'][:5]).lower()]


            # loss, acc, _, _ = model(caption, ms_desc_bins, audio_tokens, motion_tokens)
            loss, acc, _, _ = model(level,
                                    ms_desc_L, ms_des_F,
                                    vq_tokens_L, vq_tokens_F, audio_tokens)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_acc += acc

            if args.use_wandb:
                wandb.log({"batch_loss": loss.item(), "batch_acc": acc})

        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

        if args.use_wandb:
            wandb.log({"epoch_loss": avg_loss, "epoch_acc": avg_acc})

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch+1)==args.epochs:
            save_path = os.path.join(args.save_dir, f"Xmotionllm_epoch{epoch+1}.pth")
            model.save_model(save_path)
            print(f"Saved checkpoint to {save_path}")


def main():
    args = get_args_parser()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.resume_ckpt = 'ckpt/motionllm.pth'
    args.use_wandb = True
    # for pretraining
    args.task = 'caption_to_motion' # 'leader_to_follower' # 'follower_to_leader'
    args.wandb_project = "Salsa-LLM"
    args.wandb_run_name = "pretrain_all" if not args.task else args.task + '_v3'

    args.save_dir = f'output_trained/{args.wandb_run_name}'
    os.makedirs(args.save_dir, exist_ok=True)
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    model = MotionLLM(args)
    # Loading pretrained model for instruction fine-tuning

    args.resume_ckpt = 'output_trained\pretrain_all/Xmotionllm_epoch5.pth'
    if args.resume_ckpt and args.task:
        model.load_model(args.resume_ckpt)

    model.to(args.device)

    # You should implement this dataset to match your caption-motion pair format -- Done!

    train_dataset = Salsa_Dataset(args,
                    lmdb_dir='utils/salsa_utils/Salsa_Temp/lmdb_Salsa_pair/lmdb_train',
                    n_poses=100,
                    subdivision_stride=50,
                    pose_resampling_fps=20)
    args.batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    args.save_every = 1
    args.epochs = 100
    args.lr = 1e-4
    train(model, train_loader, args)


if __name__ == '__main__':
    os.chdir('S:\Payam\Dance_Salsa_SFU\Motion-Agent-Salsa')
    main()
