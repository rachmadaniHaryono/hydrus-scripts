#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""personal hydrus scripts
"""
import click
import yaml

__author__ = """rachmadani haryono"""
__email__ = """foreturiga@gmail.com"""
__version__ = """0.1.0"""


@click.group()
@click.version_option(__version__)
@click.pass_context
@click.argument("config_yaml", type=click.Path(exists=True))
@click.option("--debug/--no-debug", default=False)
def main(ctx, config_yaml, debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")
    if debug:
        click.echo(click.format_filename(config_yaml))
    with open(config_yaml) as f:
        config = yaml.safe_load(f)
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    ctx.obj["DEBUG"] = debug
    ctx.obj["CONFIG"] = config


@main.command()
@click.pass_context
def print_access_key(ctx):
    try:
        click.echo(f"access_key: {ctx.obj['CONFIG']['access_key']}")
    except Exception:
        click.echo("error:no access_key")


if __name__ == "__main__":
    main()
