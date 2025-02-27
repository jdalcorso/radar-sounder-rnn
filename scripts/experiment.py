import time
import os
import scripting
import sh


def modify_yaml(script_dir, script, parameter_name, parameter_value):
    sh.sed(
        "-i",
        "s|"
        + parameter_name
        + ": .*|"
        + parameter_name
        + ": "
        + str(parameter_value)
        + "|g",
        os.path.join(script_dir, "config_files", script + ".yaml"),
    )


def main(
    train_script,
    script_dir,
    log_dir,
    container_name,
    model,
    hidden_size,
    pos_enc,
    patch_len,
    seq_len,
    split,
    first_only,
    seed,
    epochs,
    batch_size,
    lr,
    dataset,
    log_every,
    out_dir,
    **kwargs,
):

    # Choose whether to test a fully or weakly supervised model
    match train_script:
        case "train":
            test_script = "test"
        case _:
            test_script = "test_weak"

    # Modify .yaml according to parameters
    for script in [train_script, test_script]:
        modify_yaml(script_dir, script, "model", model)
        modify_yaml(script_dir, script, "hidden_size", hidden_size)
        modify_yaml(script_dir, script, "pos_enc", pos_enc)
        modify_yaml(script_dir, script, "patch_len", patch_len)
        modify_yaml(script_dir, script, "seq_len", seq_len)
        modify_yaml(script_dir, script, "split", split)
        modify_yaml(script_dir, script, "first_only", first_only)
        modify_yaml(script_dir, script, "epochs", epochs)
        modify_yaml(script_dir, script, "batch_size", batch_size)
        modify_yaml(script_dir, script, "lr", lr)
        modify_yaml(script_dir, script, "dataset", dataset)
        modify_yaml(script_dir, script, "log_every", log_every)
        modify_yaml(script_dir, script, "out_dir", out_dir)

    # Run seeded training
    for s in range(seed):
        print("Executing experiment with seed:", s)
        modify_yaml(script_dir, train_script, "seed", s)
        modify_yaml(script_dir, test_script, "seed", s)
        sh.python(
            os.path.join(script_dir, train_script + ".py"),
            "-c",
            os.path.join(script_dir, "config_files", train_script + ".yaml"),
        )
        sh.python(
            os.path.join(script_dir, test_script + ".py"),
            "-c",
            os.path.join(script_dir, "config_files", test_script + ".yaml"),
        )


if __name__ == "__main__":
    scripting.logged_main(
        "Training",
        main,
    )
