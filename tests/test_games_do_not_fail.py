# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import os
import pathlib
import shutil
import sys


def run_game(game, params):
    dev_null_file = open(os.devnull, "w")
    old_stdout, sys.stdout = sys.stdout, dev_null_file

    game = importlib.import_module(game)
    params_as_list = [f"--{k}={v}" for k, v in params.items()]
    game.main(params_as_list)

    sys.std_out = old_stdout


def test_simple_autoenc():
    run_game(
        "egg.zoo.simple_autoenc.train",
        dict(vocab_size=3, n_features=6, n_epoch=1, max_len=2),
    )


def test_objects_game():
    run_game(
        "egg.zoo.objects_game.train",
        dict(
            perceptual_dimensions="[4, 4, 4, 4, 4]",
            vocab_size=3,
            n_distractors=1,
            n_epochs=1,
            max_len=1,
            train_samples=100,
            validation_samples=100,
            test_samples=100,
        ),
    )


def test_basic_games_game():
    run_game(
        "egg.zoo.basic_games.play",
        dict(
            mode="gs",
            train_data="egg/zoo/basic_games/data_generation_scripts/example_reconstruction_input.txt",
            validation_data="egg/zoo/basic_games/data_generation_scripts/example_reconstruction_input.txt",
            n_attributes=5,
            n_values=3,
            batch_size=50,
            max_len=4,
            vocab_size=100,
            n_epochs=1,
        ),
    )


def test_mnist_autoenc():
    run_game(
        "egg.zoo.mnist_autoenc.train", dict(vocab_size=3, n_epochs=1, batch_size=16)
    )


def test_channel():
    run_game(
        "egg.zoo.channel.train", dict(vocab_size=3, n_features=5, n_epoch=1, max_len=2)
    )


def test_compo_generalization():
    run_game(
        "egg.zoo.compo_vs_generalization.train",
        dict(
            n_values=3,
            n_epochs=1,
            n_attributes=5,
            vocab_size=200,
            max_len=2,
            batch_size=5120,
            random_seed=1,
        ),
    )


def test_compo_generalization_ood():
    run_game(
        "egg.zoo.compo_vs_generalization_ood.train",
        dict(
            n_values=10,
            n_epochs=1,
            n_attributes=2,
            vocab_size=20,
            max_len=2,
            batch_size=5120,
            random_seed=1,
            sender="ModifSender",
            receiver="ModifReceiver",
        ),
    )
    run_game(
        "egg.zoo.compo_vs_generalization_ood.learning_alone.train",
        dict(
            n_values=10,
            n_epochs=1,
            n_attributes=2,
            vocab_size=20,
            max_len=5,
            batch_size=5120,
            random_seed=1,
            archpart="sender",
            model="OrigSenderDeterministic",
            hidden=30,
        ),
    )


def test_compositional_efficiency():
    run_game(
        "egg.zoo.compositional_efficiency.discrete",
        dict(n_a=2, n_v=11, n_epochs=1, language="identity"),
    )
    run_game(
        "egg.zoo.compositional_efficiency.continuous", dict(vocab_size=5, n_epochs=1)
    )


def test_language_bottleneck():
    run_game(
        "egg.zoo.language_bottleneck.guess_number.train", dict(n_epochs=1, vocab_size=5)
    )


def test_mnist_vae():
    run_game("egg.zoo.mnist_vae.train", dict(n_epochs=1, vocab_size=5))
    dump_folder = pathlib.Path.cwd() / "dump"
    if dump_folder.is_dir():
        shutil.rmtree(dump_folder)
