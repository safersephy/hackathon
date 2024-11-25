from pathlib import Path
import tomllib

def config():
    configfile = Path("src\Leendert\config.toml")
    with configfile.open("rb") as f:
        tomlconfig = tomllib.load(f)

    assert tomlconfig["dev"] != "dev", ValueError("Please set dev in config.toml to your own name")
    assert tomlconfig["port"] != "none", ValueError("Please set port in config.toml to your own port")
    uri = tomlconfig["mlflow_uri"] + ":" + tomlconfig["port"]
    dev = tomlconfig["dev"]
    print(f"Using {uri} as mlfow uri")
    print(f"Using {dev} as dev name")
    
    return uri, dev

