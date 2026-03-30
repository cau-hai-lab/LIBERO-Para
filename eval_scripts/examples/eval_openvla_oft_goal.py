"""OpenVLA-OFT (Goal) — thin wrapper around eval_openvla_oft.py"""
import sys
sys.argv += ["--pretrained_checkpoint", "moojink/openvla-7b-oft-finetuned-libero-goal"]
if "--output_dir" not in sys.argv:
    sys.argv += ["--output_dir", "./logs_para/openvla-oft-goal/"]
from eval_openvla_oft import main
main()
