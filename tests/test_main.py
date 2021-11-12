#!/usr/bin/env python

from click.testing import CliRunner  # type:ignore

from hydrus_scripts import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert result.output


def test_split_images():
    runner = CliRunner()
    result = runner.invoke(
        main, ["split-images", "./config.yaml", "1", "--width", "1", "--height", "1"]
    )
    assert result.exit_code == 1, result.output
    assert result.output
