import optuna
import scripting
from train import main as train_main
from cycle import main as cycle_main
from cycle_mod import main as cycle_mod_main
from cycle_double import main as cycle_double_main


def main_fn(
    n_trials,
    train_func,
    model,
    hidden_size,
    patch_len,
    seq_len,
    chunk_len,
    split,
    first_only,
    data_aug,
    seed,
    epochs,
    batch_size,
    batch_number,
    lr,
    wd,
    log_every,
    log_last,
    dataset,
    out_dir,
    **kwargs,
):
    config = {
        "model": model,
        "hidden_size": hidden_size,
        "patch_len": patch_len,
        "seq_len": seq_len,
        "chunk_len": chunk_len,
        "split": split,
        "first_only": first_only,
        "data_aug": data_aug,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "batch_number": batch_number,
        "lr": lr,
        "wd": wd,
        "log_every": log_every,
        "log_last": log_last,
        "dataset": dataset,
        "out_dir": out_dir,
    }
    study = optuna.create_study(direction="minimize")
    match train_func:
        case "cycle":
            study.optimize(
                lambda trial: objective_cycle(trial, config), n_trials=n_trials
            )
        case "cycle_mod":
            study.optimize(
                lambda trial: objective_cyclemod(trial, config), n_trials=n_trials
            )
        case "cycle_double":
            study.optimize(
                lambda trial: objective_cycledouble(trial, config), n_trials=n_trials
            )
        case "train":
            study.optimize(
                lambda trial: objective_sup(trial, config), n_trials=n_trials
            )
        case _:
            raise ValueError(f"Unknown training function: {train_func}")
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)

    top_trials = sorted(study.trials, key=lambda t: t.value)[:5]
    print("Top 5 best hyperparameter configurations:")
    for i, trial in enumerate(top_trials):
        print(f"\nRank {i+1}:")
        print(f"  Loss: {trial.value}")
        print(f"  Hyperparameters: {trial.params}")


def objective_cycle(trial, config):
    return cycle_main(
        model=config["model"],
        hidden_size=config["hidden_size"],
        patch_len=config["patch_len"],
        seq_len=config["seq_len"],
        chunk_len=config["chunk_len"],
        split=config["split"],
        first_only=config["first_only"],
        seed=config["seed"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        batch_number=config["batch_number"],
        lr=trial.suggest_float("lr", config["lr"][0], config["lr"][1], log=True),
        wd=trial.suggest_float("wd", config["wd"][0], config["wd"][1], log=True),
        log_every=config["log_every"],
        log_last=config["log_last"],
        dataset=config["dataset"],
        out_dir=config["out_dir"],
    )


def objective_cyclemod(trial, config):
    return cycle_mod_main(
        model=config["model"],
        hidden_size=config["hidden_size"],
        patch_len=config["patch_len"],
        seq_len=config["seq_len"],
        chunk_len=config["chunk_len"],
        split=config["split"],
        first_only=config["first_only"],
        seed=config["seed"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        batch_number=config["batch_number"],
        lr=trial.suggest_float("lr", config["lr"][0], config["lr"][1], log=True),
        wd=trial.suggest_float("wd", config["wd"][0], config["wd"][1], log=True),
        log_every=config["log_every"],
        log_last=config["log_last"],
        dataset=config["dataset"],
        out_dir=config["out_dir"],
    )


def objective_cycledouble(trial, config):
    return cycle_double_main(
        model=config["model"],
        hidden_size=config["hidden_size"],
        patch_len=config["patch_len"],
        seq_len=config["seq_len"],
        chunk_len=config["chunk_len"],
        split=config["split"],
        first_only=config["first_only"],
        seed=config["seed"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        batch_number=config["batch_number"],
        lr=trial.suggest_float("lr", config["lr"][0], config["lr"][1], log=True),
        wd=trial.suggest_float("wd", config["wd"][0], config["wd"][1], log=True),
        log_every=config["log_every"],
        log_last=config["log_last"],
        dataset=config["dataset"],
        out_dir=config["out_dir"],
    )


def objective_sup(trial, config):
    return train_main(
        model=config["model"],
        hidden_size=config["hidden_size"],
        patch_len=config["patch_len"],
        seq_len=config["seq_len"],
        chunk_len=config["chunk_len"],
        split=config["split"],
        first_only=config["first_only"],
        data_aug=config["data_aug"],
        seed=config["seed"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=trial.suggest_float("lr", config["lr"][0], config["lr"][1], log=True),
        wd=trial.suggest_float("wd", config["wd"][0], config["wd"][1], log=True),
        log_every=config["log_every"],
        log_last=config["log_last"],
        dataset=config["dataset"],
        out_dir=config["out_dir"],
    )


if __name__ == "__main__":
    scripting.logged_main(
        "Train",
        main_fn,
    )
