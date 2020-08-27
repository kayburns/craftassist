"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import torch
import time
import training_utils as tu


def train_epoch(tms, DL, opts):
    l = 0
    error = 0
    count = 0
    dlit = iter(DL)
    tu.set_modules(tms, train=True)
    for i in range(len(DL)):
        b = dlit.next()
        targets, scores = tu.get_scores_from_datapoint(tms, b, opts)
        loss = tms["lfn"]([scores, targets])
        max_ind = torch.argmax(scores, dim=1)
        num_correct = sum(max_ind.eq(targets)).item()
        error += opts["batchsize"] - num_correct
        loss.backward()
        tms["optimizer"].step()
        l = l + loss.detach().item()
        count = count + 1
    return (l / count, error / (count * opts["batchsize"]))


def run_visualization(
    vis, tms, opts, checkpoint_path=None, num_examples=2, tsleep=1, loadpath=False
):
    if loadpath and checkpoint_path is not None and checkpoint_path != "":
        new_tms = tu.get_context_segment_trainer_modules(
            opts, checkpoint_path=checkpoint_path, backup=False, verbose=True
        )
    else:
        new_tms = tms
    tu.set_modules(new_tms, train=False)
    vis.set_model(new_tms)
    for n in range(num_examples):
        vis.visualize(use_model=True)
    time.sleep(tsleep)


if __name__ == "__main__":
    parser = tu.get_train_parser()
    opts = vars(parser.parse_args())

    # Setup the data, models and optimizer
    dataset, dataloader = tu.setup_dataset_and_loader(opts)
    tms = tu.get_context_segment_trainer_modules(
        opts, opts["checkpoint"], backup=opts["backup"], verbose=True
    )
    # The context and seg net were already moved
    if opts["cuda"] == 1:
        tms["score_module"].cuda()
        tms["lfn"].cuda()

    # Setup visualization
    vis = None
    if opts["visualize_epochs"]:
        from visualization_utils import GeoscorerDatasetVisualizer

        vis = GeoscorerDatasetVisualizer(dataset)
        vis.set_model(tms, opts)
        run_visualization(
            vis, tms, opts, checkpoint_path=None, num_examples=2, tsleep=1, loadpath=False
        )

    # Run training
    for i in range(opts["nepoch"]):
        train_loss, train_error = train_epoch(tms, dataloader, opts)
        tu.pretty_log(
            "train loss {:<5.4f} error {:<5.2f} {}".format(train_loss, train_error * 100, i)
        )
        if opts["checkpoint"] != "":
            metadata = {"epoch": i, "train_loss": train_loss, "train_error": train_error}
            tu.save_checkpoint(tms, metadata, opts, opts["checkpoint"])
        if opts["visualize_epochs"]:
            run_visualization(vis, tms, opts, opts["checkpoint"], 2, 1, False)
