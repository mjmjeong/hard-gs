import wandb
import pdb

class wandblogger(object):
    def __init__(self, args, sync_tensorboard):
        args.wandb_id = wandb.util.generate_id() if args.wandb_id is None else args.wandb_id
        #if sync_tensorboard: 
        #    wandb.tensorboard.patch(root_logdir=args.model_path)
        wandb_tags = []
        if len(args.wandb_tags) > 0:
            str_ = ''.join(args.wandb_tags)
            wandb_tags = str_.split('/')
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_name, id=args.wandb_id, resume="allow", dir=args.model_path, sync_tensorboard=sync_tensorboard, tags=wandb_tags)
        wandb.config.update(args, allow_val_change=True)

    def finish(self):
        wandb.finish()

    def log_dicts(self, dicts, step):
        wandb.log(dicts, step=step)
    
    def log_images(self, log_dict, step):
        print("log image to wandb")
        image_log = {}
        for name, dicts in log_dict.items():
            image_log[name] = wandb.Image(dicts['image'], caption=dicts['caption'])
        wandb.log(image_log, step=step)
        
