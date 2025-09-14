import asyncio
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from econagents_ibex_tudelft import run_experiment_from_yaml
from create_game import create_game_from_specs

load_dotenv()

HOSTNAME = os.getenv("HOSTNAME")
PORT = os.getenv("PORT")
USERNAME = os.getenv("GAME_USERNAME")
PASSWORD = os.getenv("GAME_PASSWORD")

# Paths to the latest spec and config
PARSE_OUT_DIR = Path(__file__).parent / "output" / "parse_out"
INTERPRET_OUT_DIR = Path(__file__).parent / "output" / "interpret_out"

# Find the latest JSON spec and YAML config
json_specs_file_path = 'output/harberger.json'
with open(json_specs_file_path, 'r') as f:
    json_specs = [Path(json_specs_file_path)]

yaml_configs = sorted(INTERPRET_OUT_DIR.glob("*.yaml"), key=os.path.getmtime, reverse=True)

if not json_specs:
    raise FileNotFoundError("No JSON spec found in output/parse_out/")
if not yaml_configs:
    raise FileNotFoundError("No YAML config found in output/interpret_out/")

LATEST_JSON_SPEC = json_specs[0]
LATEST_YAML_CONFIG = yaml_configs[0]

async def main():
    """Main function to run the game."""
    new_game_data = create_game_from_specs(
        specs_path=LATEST_JSON_SPEC,
        base_url=f"http://{HOSTNAME}",
        game_name=f"econagents_game_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        credentials={"username": USERNAME, "password": PASSWORD},
    )
    game_id = new_game_data["game_id"]
    login_payloads = new_game_data["login_payloads"]

    await run_experiment_from_yaml(LATEST_YAML_CONFIG, login_payloads, game_id=game_id)

if __name__ == "__main__":
    asyncio.run(main())

