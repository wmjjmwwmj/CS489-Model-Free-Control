import os
import argparse
from datetime import datetime
import gym

from agent import *

parser = argparse.ArgumentParser(description="parameter setting for mujoco")
parser.add_argument('--env_name', type=str, default="Hopper-v2")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--method', choices=['ppo', 'sac'], default='sac')
parser.add_argument('--is_per', type=bool, default=True)
parser.add_argument('--entropy_tuning', type=bool, default=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if args.seed:
    fix_seed(args.seed)

env = gym.make(args.env_name)
log_dir = os.path.join('local', args.env_name,
   f'{args.method}-seed{args.seed}-per{args.is_per}-tune{args.entropy_tuning}-{datetime.now().strftime("%Y%m%d-%H%M")}')

agent = SACAgent(env, args.entropy_tuning, args.is_per, log_dir) if args.method == 'sac' else PPOAgent(env, log_dir)
agent.run()