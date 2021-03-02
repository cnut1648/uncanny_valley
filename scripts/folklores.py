import os
from pathlib import Path

from config import DATASET_DIR
from utils.types import FolkLoreData


def getTextFolklore(path: Path) -> FolkLoreData:
    """
    get text path from directories
    not loading so that save memory
    """
    folklore_data = {}
    for book in os.listdir(path):
        text_for_book = []
        for text in os.listdir(path / book):
            if text.lower().startswith("split "):
                text_for_book = [text]
                continue
            if text != ".ipynb_checkpoints":
                text_for_book.append(path / book / text)

        # postprocess book that has split by
        first = text_for_book[0]
        if str(first).lower().startswith("split "):
            text_for_book = [path / book / first / file for file in os.listdir(path / book / first) if
                             file != ".ipynb_checkpoints"]
        folklore_data[book] = text_for_book

    return folklore_data


if __name__ == "__main__":
    folklore_data = getTextFolklore(DATASET_DIR)

    folklore_data.update(getTextFolklore(DATASET_DIR / "The King James Version of the Bible"))
    del folklore_data["The King James Version of the Bible"]
    # TODO
    del folklore_data["Folklore and Mythology Electronic Texts"]

    for book in folklore_data:
        print(book)
        print("\t", [str(os.path.basename(p)) for p in folklore_data[book]])
