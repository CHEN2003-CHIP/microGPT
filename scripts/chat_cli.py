"""Command-line chat for the microGPT teaching project."""

import argparse
import sys


parser = argparse.ArgumentParser(description="Chat with a trained model")
parser.add_argument("-i", "--source", type=str, default="sft", choices=["base", "sft"], help="Checkpoint source")
parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
parser.add_argument("-s", "--step", type=int, default=None, help="Checkpoint step to load")
parser.add_argument("-p", "--prompt", type=str, default="", help="Single prompt mode")
parser.add_argument("-t", "--temperature", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("-k", "--top-k", type=int, default=50, help="Top-k sampling")
parser.add_argument("-m", "--max-tokens", type=int, default=256, help="Maximum generated tokens per answer")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device override")
args = parser.parse_args()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from microchat.checkpoint_manager import load_model
from microchat.common import autodetect_device_type, compute_init
from microchat.engine import Engine


device_type = autodetect_device_type() if args.device_type == "" else args.device_type
_, _, _, _, device = compute_init(device_type)
model, tokenizer, metadata = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
engine = Engine(model, tokenizer)

bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")

print("\nmicroGPT CLI")
print("-" * 40)
print("type 'quit' to exit")
print("type 'clear' to reset the conversation")
print("-" * 40)

conversation_tokens = [bos]
while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if user_input.lower() in {"quit", "exit"}:
        print("Goodbye!")
        break
    if user_input.lower() == "clear":
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue
    if not user_input:
        continue

    conversation_tokens.extend([user_start, *tokenizer.encode(user_input), user_end, assistant_start])
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    for token_column, _ in engine.generate(
        conversation_tokens,
        num_samples=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ):
        token = token_column[0]
        if token == assistant_end:
            break
        response_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
    print()

    if not response_tokens or response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    if args.prompt:
        break
