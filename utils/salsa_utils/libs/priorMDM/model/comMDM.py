import torch
import torch.nn as nn

from model.mdm import MDM

class ComMDM(MDM):

    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, args=None, **kargs):

        super(ComMDM, self).__init__(modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                         latent_dim, ff_size, num_layers, num_heads, dropout,
                         ablation, activation, legacy, data_rep, dataset, clip_dim,
                         arch, emb_trans_dec, clip_version, **kargs)
        self.is_multi = True
        self.args = args

        self.multi_person = MultiPersonBlock(arch=self.args.multi_arch,
                                             fn_type=self.args.multi_func,
                                             num_layers=self.args.multi_num_layers,
                                             latent_dim=self.latent_dim,
                                             input_feats=self.input_feats,
                                             predict_6dof=self.args.predict_6dof)
        self.multi_person = MultiPersonBlock_Salsa(arch=self.args.multi_arch,
                                             fn_type=self.args.multi_func,
                                             num_layers=self.args.multi_num_layers,
                                             latent_dim=self.latent_dim,
                                             input_feats=self.input_feats,
                                             predict_6dof=self.args.predict_6dof)

        if self.arch == 'trans_enc':
            assert 0 < self.args.multi_backbone_split <= self.num_layers
            print(f'CUTTING BACKBONE AT LAYER [{self.args.multi_backbone_split}]')
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            del self.seqTransEncoder
            self.seqTransEncoder_start = nn.TransformerEncoder(seqTransEncoderLayer,
                                                               num_layers=self.args.multi_backbone_split)
            self.seqTransEncoder_end = nn.TransformerEncoder(seqTransEncoderLayer,
                                                             num_layers=self.num_layers - self.args.multi_backbone_split)
        else:
            raise ValueError('Supporting only trans_enc arch.')

        if self.args.multi_mdm_freeze:
            self.freeze_block(self.input_process)
            self.freeze_block(self.sequence_pos_encoder)
            self.freeze_block(self.seqTransEncoder_start)
            self.freeze_block(self.seqTransEncoder_end)
            self.freeze_block(self.embed_timestep)
            if 'text' in self.cond_mode:
                self.freeze_block(self.embed_text)
            self.freeze_block(self.output_process)

    def forward(self, x, timesteps, y=None):
        canon, x = torch.split(x,[1, x.shape[-1]-1], dim=-1)
        canon_other, x_other = torch.split(y['other_motion'],[1, y['other_motion'].shape[-1]-1], dim=-1)

        # Build embedding vector
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        force_no_com = y.get('no_com', False)  # FIXME - note that this feature not working for com_only - which is ok
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode: # Todo: we may use this to inject VQ tokens
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)
        # Todo-Payam: this is where we should include audio or motion tokens.
        # Embed motion to latent space (frame by frame)
        x = self.input_process(x) #[seqlen, bs, d]
        x_other = self.input_process(x_other)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

        x_other = torch.cat((emb, x_other), axis=0)
        x_other = self.sequence_pos_encoder(x_other)


        mid = self.seqTransEncoder_start(xseq)
        mid_other = self.seqTransEncoder_start(x_other)
        cur_canon = canon if self.args.predict_6dof else None
        other_canon = canon_other if self.args.predict_6dof else None

        # delta_x, canon_out = self.multi_person(cur=mid, other=mid_other, cur_canon=cur_canon,
        #                                        other_canon=other_canon)
        delta_x, canon_out = self.multi_person(p1_canon=cur_canon, p1_motion=mid, p1_vq_token=y.get('vq_token_input', False),
                                               p2_canon=other_canon, p2_motion=mid_other, p2_vq_token=y.get('vq_token_other', False))


        if force_no_com:
            output = self.seqTransEncoder(xseq)[1:]  # [seqlen, bs, d]
        else:
            if 'out_cur' in self.multi_person.fn_type:
                mid += delta_x
            elif 'out_cond' in self.multi_person.fn_type:
                mid[0] += delta_x[0]

            if self.args.multi_backbone_split < self.num_layers:
                output = self.seqTransEncoder_end(mid)[1:]
            elif self.args.multi_backbone_split == self.num_layers:
                output = mid[1:]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        output = torch.cat((canon_out, output), dim=-1)
        return output

    def trainable_parameters(self):
        return [p for name, p in self.named_parameters() if p.requires_grad]
        # return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def multi_parameters(self):
        return [p for name, p in self.multi_person.named_parameters() if p.requires_grad]

    def freeze_block(self, block):
        block.eval()
        for p in block.parameters():
            p.requires_grad = False



class MultiPersonBlock(nn.Module):
    def __init__(self, arch, fn_type, num_layers, latent_dim, input_feats, predict_6dof):
        super().__init__()
        self.arch = arch
        self.fn_type = fn_type
        self.predict_6dof = predict_6dof
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_heads = 4
        self.ff_size = 1024
        self.dropout = 0.1
        self.activation = 'gelu'
        self.input_feats = input_feats
        if self.predict_6dof:
            self.canon_agg = nn.Linear(4*2, self.latent_dim)
            # self.canon_agg = nn.Linear(self.input_feats*2, self.latent_dim)
            self.canon_out = nn.Linear(self.latent_dim, 4)
            # self.canon_out = nn.Linear(self.latent_dim, self.input_feats)
        if 'in_both' in self.fn_type:
            self.aggregation = nn.Linear(self.latent_dim*2, self.latent_dim)
        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            self.model = nn.TransformerEncoder(seqTransEncoderLayer,
                                               num_layers=self.num_layers)
        else:
            raise NotImplementedError()

    def forward(self, other, cur=None, cur_canon=None, other_canon=None):

        if 'in_both' in self.fn_type:
            assert other is not None
            x = self.aggregation(torch.concatenate((cur, other), dim=-1))
        else:
            x = other

        if self.predict_6dof:
            assert cur_canon is not None and other_canon is not None
            cur_canon = cur_canon.squeeze(-1).permute(2, 0, 1)[..., :4]
            other_canon = other_canon.squeeze(-1).permute(2, 0, 1)[..., :4]
            canon = self.canon_agg(torch.concatenate((cur_canon, other_canon), dim=-1))
            x = torch.concatenate((canon, x), dim=0)

        out = self.model(x) # 198 goes in
        if self.predict_6dof:
            canon, out = torch.split(out, [1, out.shape[0] - 1], dim=0)
            canon = self.canon_out(canon).permute(1, 2, 0).unsqueeze(-1)
            pad = torch.zeros([canon.shape[0], 263-4, 1, 1], device=canon.device, dtype=canon.dtype)
            canon = torch.cat((canon, pad), axis=1)
        else:
            canon = None

        return out, canon



class MultiPersonBlock_Salsa(nn.Module):
    def __init__(self, arch, fn_type, num_layers, latent_dim, input_feats, predict_6dof):
        super().__init__()
        self.arch = arch
        self.fn_type = fn_type
        self.predict_6dof = predict_6dof
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_heads = 4
        self.ff_size = 1024
        self.dropout = 0.1
        self.activation = 'gelu'
        self.input_feats = input_feats
        if self.predict_6dof:
            self.canon_agg = nn.Linear(4*2, self.latent_dim)
            # self.canon_agg = nn.Linear(self.input_feats*2, self.latent_dim)
            self.canon_out = nn.Linear(self.latent_dim, 4)
            # self.canon_out = nn.Linear(self.latent_dim, self.input_feats)
        if 'in_both' in self.fn_type:
            self.aggregation = nn.Linear(self.latent_dim*2, self.latent_dim)

        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            self.model = nn.TransformerEncoder(seqTransEncoderLayer,
                                               num_layers=self.num_layers)

            self.vq_tokens_nb = 512
            self.vq_proj = nn.Embedding(self.vq_tokens_nb, self.latent_dim)  # adjust if needed


            self.vq_cross_block = CrossAttentionBlock(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dropout=self.dropout,
                ff_size=self.ff_size,
                activation=self.activation
            )

            self.inter_cross_block = CrossAttentionBlock(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dropout=self.dropout,
                ff_size=self.ff_size,
                activation=self.activation
            )


        else:
            raise NotImplementedError()

    # def forward(self, other, cur=None, cur_canon=None, other_canon=None):
    def forward(self, p1_canon, p1_motion, p1_vq_token, p2_canon, p2_motion, p2_vq_token):

        # # Step 1: Feature aggregation if both are used
        # if 'in_both' in self.fn_type:
        #     assert p2_motion is not None
        #     x = self.aggregation(torch.concatenate((p1_motion, p2_motion), dim=-1))
        # else:
        #     x = p2_motion
        #
        # # Step 2: Canonical feature concatenation (if used)
        # if self.predict_6dof:
        #     assert p1_canon is not None and p2_canon is not None
        #     cur_canon = p1_canon.squeeze(-1).permute(2, 0, 1)[..., :4]
        #     other_canon = p2_canon.squeeze(-1).permute(2, 0, 1)[..., :4]
        #     canon = self.canon_agg(torch.concatenate((cur_canon, other_canon), dim=-1))
        #     x = torch.concatenate((canon, x), dim=0)
        #
        # # Step 3: Temporal encoding with TransformerEncoder
        # out = self.model(x) # 198 goes in
        #
        # # Step 4: Strip canon from sequence and decode 6DOF
        # if self.predict_6dof:
        #     canon, out = torch.split(out, [1, out.shape[0] - 1], dim=0)
        #     canon = self.canon_out(canon).permute(1, 2, 0).unsqueeze(-1)
        #     pad = torch.zeros([canon.shape[0], 263-4, 1, 1], device=canon.device, dtype=canon.dtype)
        #     canon = torch.cat((canon, pad), axis=1)
        # else:
        #     canon = None


        # Step 5: Apply VQ-token cross attention (optional)
        p1_vq_token_aligned = p1_vq_token.repeat_interleave(4, dim=1)
        vq_embed = self.vq_proj(p1_vq_token_aligned).permute(1, 0, 2)  # (T, B, D)
        L, B, H = vq_embed.shape
        pad = torch.zeros(p1_motion.shape[0]-L, B, H, dtype=vq_embed.dtype, device=vq_embed.device)
        vq_embed = torch.cat([vq_embed, pad])
        #     canon = torch.cat((canon, pad), axis=1)

        # updated_p1 = self.vq_cross_block(p1_motion, vq_embed)
        q = p1_motion.clone()
        updated_p1 = self.vq_cross_block(q, vq_embed)

        # Step 6: Apply two-person cross attention (optional)
        updated_p1 = self.inter_cross_block(updated_p1, p2_motion)

        # Step 3: Self-attention (temporal refinement)
        updated_p1 = self.model(updated_p1)


        delta = updated_p1 - p1_motion
        canon = p1_canon
        return delta, canon



import torch.nn.functional as F
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, ff_size=1024, activation='gelu'):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, ff_size)
        self.activation = F.gelu if activation == 'gelu' else F.relu
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_size, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, context):
        # x, context: (T, B, D)
        attn_out, _ = self.cross_attn(query=x, key=context, value=context)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout3(ffn_out))

        return x