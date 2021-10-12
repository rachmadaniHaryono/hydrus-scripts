#!/usr/bin/env python3

"""personal hydrus scripts
"""
import collections
import json
import logging
import pprint
import re
import timeit
import typing as T
from urllib.parse import urlparse

import click
import hydrus
import more_itertools
import tqdm
import yaml

__author__ = """rachmadani haryono"""
__email__ = """foreturiga@gmail.com"""
__version__ = """0.1.0"""


@click.group()
@click.version_option(__version__)
@click.pass_context
@click.option("--debug/--no-debug", default=False)
def main(ctx, debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    ctx.obj["DEBUG"] = debug


def load_config(config_yaml):
    if config_yaml is not None:
        with open(config_yaml) as f:
            return yaml.safe_load(f)


@main.command()
@click.argument("config-yaml", type=click.Path(exists=True))
def print_access_key(config_yaml):
    config = load_config(config_yaml)
    try:
        click.echo(f"access_key: {config['access_key']}")
    except Exception:
        click.echo("error:no access_key")


class TagChanger:
    def __init__(self, tag_dict):
        self.tag_dict = tag_dict

    @staticmethod
    def text_to_dict(text):
        """convert text to dict."""
        res = collections.defaultdict(set)
        text = "\n".join(x.strip() for x in text.splitlines())
        text = "\n\n".join(re.split("\n\n+", text))
        for item in text.split("\n\n"):
            tags = item.splitlines()
            res[tags[0]].update(tags[1:])
        return res

    def handle_fmd(self, fmd) -> T.Iterator[str]:
        """return iterator of key tag to be removed"""
        tags = fmd["service_names_to_statuses_to_display_tags"]["my tags"]["0"]
        for tag in self.tag_dict:
            if tag in tags:
                yield tag

    def handle_key(self, key, hashes):
        tag_data_dict = {"1": [key]}
        if self.tag_dict[key]:
            tag_data_dict["0"] = list(self.tag_dict[key])
        return dict(
            hashes=list(hashes),
            service_to_action_to_tags={"my tags": tag_data_dict},
        )

    def run(self, client, tags_list):
        fids = set(
            more_itertools.flatten(
                [client.search_files(list(x)) for x in tqdm.tqdm(tags_list)]
            )
        )
        key_hashes = collections.defaultdict(set)
        for chunk in tqdm.tqdm(list(more_itertools.chunked(fids, 128))):
            for fmd in client.file_metadata(file_ids=chunk):
                hash_ = fmd["hash"]
                for key in self.handle_fmd(fmd):
                    key_hashes[key].add(hash_)
        for key, hashes in tqdm.tqdm(sorted(key_hashes.items())):
            kwargs = self.handle_key(key, hashes)
            tqdm.tqdm.write("key ({}):".format(len(hashes)) + str(key))
            if kwargs:
                client.add_tags(**kwargs)


class TagChangerF2(TagChanger):
    @staticmethod
    def text_to_dict(text):
        """convert text to dict."""
        res = collections.defaultdict(set)
        text = "\n".join(x.strip() for x in text.splitlines())
        text = "\n\n".join(re.split("\n\n+", text))
        for item in text.split("\n\n"):
            parts = item.splitlines()
            main_tag = parts[0].strip()
            for subitem in parts[1:]:
                c_subitem = subitem.rsplit("(", 1)[0].strip()
                res[c_subitem].add(main_tag)
                try:
                    res[c_subitem].add(c_subitem.split(main_tag, 1)[1].strip())
                except IndexError as err:
                    logging.error(str({"main_tag": main_tag, "c_subitem": c_subitem}))
                    raise err
        return res


@main.command()
@click.argument("config-yaml", type=click.Path(exists=True))
@click.argument("tags-file", type=click.Path(exists=True))
@click.option("--mode", default=1)
def replace_tag(config_yaml, tags_file, mode):
    config = load_config(config_yaml)
    with open(tags_file) as f:
        text = f.read()
    obj_dict = {1: TagChanger, 2: TagChangerF2}
    obj_cls = obj_dict[mode]
    obj = obj_dict[mode](tag_dict=obj_cls.text_to_dict(text))
    obj.run(hydrus.Client(config["access_key"]), [[x] for x in obj.tag_dict.keys()])


@main.command()
@click.argument("config-yaml", type=click.Path(exists=True))
@click.argument("hashes-file", type=click.Path(exists=True))
def count_netloc(config_yaml, hashes_file):
    with open(hashes_file) as f:
        text = f.read()
    fmds = hydrus.Client(load_config(config_yaml)["access_key"]).file_metadata(
        hashes=text.splitlines()
    )
    known_urls = list(more_itertools.flatten([x.get("known_urls", []) for x in fmds]))
    pprint.pprint(collections.Counter([urlparse(x).netloc for x in known_urls]))


@main.command()
@click.argument("config-yaml", type=click.Path(exists=True))
@click.argument("sibling-file", type=click.Path(exists=True))
def count_sibling(config_yaml, sibling_file):
    with open(sibling_file) as f:
        lines = f.read().splitlines()
    assert len(lines) % 2 == 0
    data = []
    len_lines = len(lines)
    #  len_lines = 10
    for idx in range(len_lines // 2):
        data.append([lines[idx * 2], lines[(idx * 2) + 1]])
    tag_list = sorted({x[1] for x in data})
    client = hydrus.Client(load_config(config_yaml)["access_key"])
    max_limit = 1024
    skipped_tags = []
    if max_limit:
        fids = set()
        for tag in tqdm.tqdm(tag_list):
            cfids = client.search_files([tag])
            if len(cfids) < 1024:
                fids.update(cfids)
            else:
                tqdm.tqdm.write("skip:" + str(tag))
                skipped_tags.append(tag)

    else:
        fids = set(
            more_itertools.flatten(
                [client.search_files([x]) for x in tqdm.tqdm(tag_list)]
            )
        )
    fmds = list(
        more_itertools.flatten(
            client.file_metadata(file_ids=chunk)
            for chunk in tqdm.tqdm(list(more_itertools.chunked(fids, 128)))
        )
    )
    f_tags = [x[0] for x in data]
    count = (
        collections.Counter(
            more_itertools.flatten(
                [
                    fmd["service_names_to_statuses_to_tags"]["my tags"]["0"]
                    for fmd in fmds
                ]
            )
        ).most_common(),
    )
    pprint.pprint(list(filter(lambda x: x[0] in f_tags, count)))
    print("No count:")
    for item in data:
        if item[0] in f_tags and item[1] not in skipped_tags:
            print(item[0])


@main.command()
@click.argument("config-yaml", type=click.Path(exists=True))
@click.argument("search-tag")
@click.option("--max-count", default=10)
def analyze_tags(config_yaml, search_tag, max_count=10):
    client = hydrus.Client(load_config(config_yaml)["access_key"])
    fids = client.search_files([search_tag])
    print(len(fids))
    fmds = list(
        more_itertools.flatten(
            client.file_metadata(file_ids=chunk)
            for chunk in tqdm.tqdm(list(more_itertools.chunked(fids, 128)))
        ),
    )
    fmds_tags = list(
        filter(
            lambda x: not x.startswith(
                ("dd:", "series:", "category:genre:", "filename:", "referer:")
            ),
            more_itertools.flatten(
                [
                    fmd["service_names_to_statuses_to_display_tags"]["my tags"]["0"]
                    for fmd in fmds
                ]
            ),
        )
    )
    pprint.pprint(
        sorted(
            collections.Counter(tuple(x.split()[:2]) for x in fmds_tags).most_common(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_count]
    )
    pprint.pprint(
        sorted(
            collections.Counter(
                tuple(x.split()[0][:2]) for x in fmds_tags
            ).most_common(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_count]
    )
    pprint.pprint(
        sorted(
            collections.Counter(
                tuple(x.split()[0][:3]) for x in fmds_tags
            ).most_common(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_count]
    )


@main.command()
@click.argument("tag-file", type=click.Path(exists=True))
@click.option("--max-count", default=10)
def analyze_local_tags(tag_file, max_count=10):
    with open(tag_file) as f:
        tags = [x.rsplit("(", 1)[0].strip() for x in f.read().splitlines()]
    pprint.pprint(
        sorted(
            collections.Counter(
                more_itertools.flatten([x.split() for x in tags])
            ).most_common(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_count]
    )
    new_tags = []
    for tag in [x for x in tags if x]:
        for idx in range(len(tag.split())):
            new_tag = " ".join(tag.split()[: idx + 1])
            if len(new_tag) > 1:
                new_tags.append(new_tag)
    most_common = collections.Counter(new_tags).most_common()
    pprint.pprint(
        sorted(
            most_common,
            key=lambda x: x[1],
            reverse=True,
        )[:max_count]
    )


@main.command()
@click.argument("config-yaml", type=click.Path(exists=True))
@click.argument("sibling-file")
def analyze_sibling(config_yaml, sibling_file):
    with open(sibling_file) as f:
        lines = f.read().splitlines()
    assert len(lines) % 2 == 0
    data = []
    len_lines = len(lines)
    #  len_lines = 10
    for idx in range(len_lines // 2):
        data.append([lines[idx * 2], lines[(idx * 2) + 1]])
    data_dict = {k: v for k, v in data}
    for k, v in tqdm.tqdm(data_dict.items()):
        if v in data_dict:
            #  tqdm.tqdm.write(str((k, v)))
            tag = v
            while tag in data_dict:
                if tag in data_dict and data_dict[tag] != tag and data_dict[tag] != v:
                    tag = data_dict[tag]
                else:
                    break
            print("\n".join([k, v, k, tag]))


@main.command()
@click.argument("config-yaml", type=click.Path(exists=True))
@click.argument("rule-file")
@click.option("--measure", is_flag=True)
def lint_tag(config_yaml, rule_file, measure=False):
    with open(rule_file) as f:
        rules = json.load(f)
    client = hydrus.Client(load_config(config_yaml)["access_key"])
    client_kwargs_list = []
    logging.basicConfig(level=logging.INFO)
    start = None
    measure_history = []
    for rule in rules:
        if rule.get("disable", False):
            continue
        if rule.get("debug", False):
            import pdb

            pdb.set_trace()
        if measure:
            start = timeit.default_timer()
        template = rule.get("template", "")
        tags = rule.get("tags", None)
        if template == "remove_tags_from_main_tags":
            search_tags_list = rule.get("search_tags", [])
            if search_tags_list:
                logging.warning(
                    "rule with template "
                    '"remove_tags_from_main_tags" contain "search_tags" key. '
                    'use "tags" key instead'
                )
            if tags:
                search_tags_list.append(tags)
            remove_tags = rule["remove_tags"]
            search_tags_list.append(remove_tags)
        else:
            if tags:
                logging.warning('rule contain "tags" key but will not be used')
            search_tags_list = rule.get("search_tags", None)
            remove_tags = rule.get("remove_tags", None)
            if remove_tags and not search_tags_list:
                search_tags_list = remove_tags
        try:
            fids = client.search_files(search_tags_list)
        except hydrus.MissingParameter:
            logging.error("MissingParameter, rule:" + str(rules))
            continue
        ordered_rule_log = dict(
            sorted([x for x in rule.items() if x[0] != "exclude_hashes"])
        )
        if not fids:
            logging.info(f"rule (0):{ordered_rule_log}")
            continue
        client_kwargs = {
            "hashes": set(),
            "service_to_action_to_tags": {"my tags": {}},
        }
        if remove_tags:
            client_kwargs["service_to_action_to_tags"]["my tags"]["1"] = list(
                set(remove_tags)
            )
        if add_tags := rule.get("add_tags", None):
            client_kwargs["service_to_action_to_tags"]["my tags"]["0"] = list(
                set(add_tags)
            )
        for chunk in tqdm.tqdm(list(more_itertools.chunked(fids, 256))):
            for fmd in client.file_metadata(file_ids=chunk):
                if (exclude_hashes := rule.get("exclude_hashes", [])) and fmd[
                    "hash"
                ] in exclude_hashes:
                    continue
                else:
                    client_kwargs["hashes"].add(fmd["hash"])
        client_kwargs["hashes"] = list(client_kwargs["hashes"])
        client_kwargs_list.append(client_kwargs)
        logging.info(
            "rule ({}):{}".format(len(client_kwargs["hashes"]), ordered_rule_log)
        )
        if measure:
            measure_history.append((ordered_rule_log, timeit.default_timer() - start))
    if measure:
        print()
        for item in sorted(measure_history, key=lambda x: x[1]):
            print(f"{item[1]}: {item[0]}")
        print()
    for kwargs in client_kwargs_list:
        if kwargs["hashes"]:
            client.add_tags(**kwargs)
            temp_kwargs = kwargs.copy()
            len_hashes = len(temp_kwargs["hashes"])
            del temp_kwargs["hashes"]
            print(
                "\n".join(sorted(kwargs["hashes"]))
                + "\n"
                + f"count: {len_hashes}; "
                + str(temp_kwargs)
                + "\n"
            )


if __name__ == "__main__":
    main()
