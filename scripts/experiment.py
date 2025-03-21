import os
import sh

import scripting
import torch


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
    model,
    hidden_size,
    patch_len,
    seq_len,
    seq_len_test,
    split,
    first_only,
    seed,
    epochs,
    batch_size,
    lr,
    wd,
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
        modify_yaml(script_dir, script, "patch_len", patch_len)
        modify_yaml(script_dir, script, "split", split)
        modify_yaml(script_dir, script, "first_only", first_only)
        modify_yaml(script_dir, script, "epochs", epochs)
        modify_yaml(script_dir, script, "batch_size", batch_size)
        modify_yaml(script_dir, script, "lr", lr)
        modify_yaml(script_dir, script, "wd", wd)
        modify_yaml(script_dir, script, "dataset", dataset)
        modify_yaml(script_dir, script, "log_every", log_every)
        modify_yaml(script_dir, script, "out_dir", out_dir)
    modify_yaml(script_dir, train_script, "seq_len", seq_len)
    modify_yaml(script_dir, test_script, "seq_len", seq_len_test)

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

    # Process results
    print("Results of the experiment with {} seeds:".format(seed))
    n_classes = 5 if dataset == "mcords3" else 4
    precision = torch.zeros(seed, n_classes)
    recall = torch.zeros(seed, n_classes)
    f1 = torch.zeros(seed, n_classes)
    accuracy = torch.zeros(seed)
    for s in range(seed):
        seed_dict = torch.load(os.path.join(out_dir, "report_seed_" + str(s) + ".pt"))
        for c in range(n_classes):
            precision[s, c] = seed_dict[str(c)]["precision"]
            recall[s, c] = seed_dict[str(c)]["recall"]
            f1[s, c] = seed_dict[str(c)]["f1-score"]
            accuracy[s] = seed_dict["accuracy"]
    for c in range(n_classes):
        print("\nClass:", c)
        print(
            "Precision:",
            precision[:, c].mean().item(),
            "(",
            precision[:, c].std().item(),
            ")",
        )
        print(
            "Recall:", recall[:, c].mean().item(), "(", recall[:, c].std().item(), ")"
        )
        print("F1:", f1[:, c].mean().item(), "(", f1[:, c].std().item(), ")")
    print("\nAccuracy:", accuracy.mean().item(), "(", accuracy.std().item(), ")")


if __name__ == "__main__":
    scripting.logged_main(
        "Training",
        main,
    )
