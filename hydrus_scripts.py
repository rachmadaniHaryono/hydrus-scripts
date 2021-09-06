#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""personal hydrus scripts
"""
import click

__author__ = """rachmadani haryono"""
__email__ = """foreturiga@gmail.com"""
__version__ = """0.1.0"""


@click.group()
@click.version_option(__version__)
@click.option("--debug/--no-debug", default=False)
def main(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")


@main.command()
def sync():
    click.echo("Syncing")


if __name__ == "__main__":
    main()
