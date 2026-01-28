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

            if len(batch) == 8:
                level, ms_desc_L, ms_des_F, vq_tokens_L, vq_tokens_F, audio_tokens, aux_batch, batch_interhuman = batch
            else:
                level, ms_desc_L, ms_des_F, vq_tokens_L, vq_tokens_F, audio_tokens, aux_batch = batch
                batch_interhuman = None

            loss, acc, _, _ = model(level,
                                    ms_desc_L, ms_des_F,
                                    vq_tokens_L, vq_tokens_F, audio_tokens,
                                    batch_interhuman_data=batch_interhuman)

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
    # MotionLLM uses --learning-rate (llm_lr); default 1e-5 to match Motion-Agent
    args.lr = getattr(args, 'llm_lr', 1e-5)
    args.wandb_project = getattr(args, 'wandb_project', "Salsa-LLM")
    # Stage 1: task 'none' or 'all' = all tasks; use pretrain_all unless --wandb-run-name is set
    _all_tasks = (args.task in (None, 'none', 'all'))
    args.wandb_run_name = getattr(args, 'wandb_run_name', None) or ("pretrain_all" if _all_tasks else f"{args.task}_v3")
    args.save_dir = getattr(args, 'save_dir', None) or f'output_trained/{args.wandb_run_name}'
    os.makedirs(args.save_dir, exist_ok=True)
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Use HumanML3D/MDM cache when motion_repr_type is humanml3d (matches demo.py --create_cache_only)
    if getattr(args, 'motion_repr_type', 'humanml3d') == 'humanml3d':
        args.is_MDM = True
    lmdb_dir = getattr(args, 'lmdb_dir', 'dataset_processed_New/lmdb_Salsa_pair/lmdb_train')
    # n_poses, subdivision_stride, pose_resampling_fps align with demo.py and README cache creation
    n_poses, subdivision_stride, pose_resampling_fps = 100, 50, 20

    model = MotionLLM(args)
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        model.load_model(args.resume_ckpt)

    model.to(args.device)

    train_dataset = Salsa_Dataset(args,
                    lmdb_dir=lmdb_dir,
                    n_poses=n_poses,
                    subdivision_stride=subdivision_stride,
                    pose_resampling_fps=pose_resampling_fps)
    batch_size = getattr(args, 'train_batch_size', 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train(model, train_loader, args)


if __name__ == '__main__':
    # os.chdir('S:\Payam\Dance_Salsa_SFU\Motion-Agent-Salsa')  # Windows-specific path, commented for Ubuntu compatibility
    main()
