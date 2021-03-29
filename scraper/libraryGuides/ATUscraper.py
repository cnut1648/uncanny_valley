import json
from collections import defaultdict
from typing import Dict, List

from dataclasses import dataclass
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains


@dataclass
class Text:
    title: str
    origin: str
    text: str


@dataclass
class ATUText:
    atu: str
    desc: str
    texts: List[Text]


# {url: {"ATU": atu, "texts": [text1, text2], ..}
visited_texts: Dict[str, ATUText] = {}


def get_consecutive_texts(tag) -> str:
    texts = []
    if tag.parent.name == "center":
        tag = tag.parent
    for sib in tag.next_siblings:
        # either <p>xxx</p>
        if sib.name == "p" or sib.name == "blockquote":
            texts.append(sib.get_text())
        elif sib.name == "center" and sib.find("tr") is not None:
            texts.append(sib.get_text())
        # or xxx (at least 5 chars to ignore \n)
        elif sib.name is None and len(str(sib)) >= 5:
            texts.append(str(sib))
        # when h2 / hr and texts already have eles
        # if texts is empty can ignore hr
        elif sib.name in ["h2", "hr"] and texts:
            break
        # other are ignores, note if there are multiple version (version separate by <h3> and <hr>),
        # this only retain version A
    return "\n".join(texts)


def get_pitts_text(driver, debug=False):
    is_special = False
    try:
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located(
                (By.XPATH, "//h2")
            )
        )
    # when page doesn't have h2, thus special treatment
    # always belong to "type" url
    except TimeoutException as e:
        is_special = True
    bs = BeautifulSoup(driver.page_source, 'lxml')
    url = driver.current_url

    # one story per page
    if "grimm" in driver.current_url:
        title = bs.find("h1")
        origin = title.findNextSibling()
        text = get_consecutive_texts(origin)
        if not debug:
            visited_texts[url].texts.append(
                Text(
                    title=title.get_text(),
                    origin=origin.get_text(),
                    text=text
                )
            )
    # multiple text in one page
    else:
        # if "type" in url or any other type
        if is_special:
            center = bs.find("center")
            title = center.find("h1").get_text()
            origin = center.find("a").get_text()
            text = get_consecutive_texts(center)
            if not debug:
                visited_texts[url].texts.append(
                    Text(
                        title=title,
                        origin=origin,
                        text=text
                    )
                )
        else:
            # 0th is content table
            for title in bs.find_all("h2")[1:]:
                ignore_prefix = ["Links to ", "Bibliography ", "Notes and Bibliography", "Additional", "Related "]
                title_str = title.get_text()
                if all(not title_str.startswith(prefix) for prefix in ignore_prefix):
                    origin = title.findNextSibling()
                    text = get_consecutive_texts(origin)
                    if origin.name == "i" and origin.find("h3") is not None:
                        origin = origin.find("h3")
                    if not debug:
                        visited_texts[url].texts.append(
                            Text(
                                title=title_str,
                                origin=origin.get_text(),
                                text=text
                            )
                        )

def click_pitts_texts(driver):
    table = driver.find_element(By.CSS_SELECTOR, ".s-lib-box.s-lib-box-std")
    anchors = table.find_elements(By.XPATH, "//td/a")
    for anchor in anchors:
        href = anchor.get_attribute("href")
        if href.startswith("http://www.pitt.edu/") and href not in visited_texts:
            atu, desc = anchor.text.split(" ", 1)
            visited_texts[href] = ATUText(atu=atu, desc=desc, texts=[])
            main_page = driver.window_handles[0]
            anchor.click()
            driver.switch_to.window(driver.window_handles[1])

            get_pitts_text(driver)

            driver.close()
            driver.switch_to.window(main_page)


def export_jsonl():
    with open("ATU.jl", "w") as f:
        for url, atuText in visited_texts.items():
            for text in atuText.texts:
                # ignore broken ones (without text)
                if text.text:
                    f.write(json.dumps({
                        "atu": atuText.atu.strip(),
                        "desc": atuText.desc.strip(),
                        "title": text.title.strip(),
                        "origin": text.origin.strip(),
                        "text": text.text.strip(),
                        "url": url.strip(),
                    }) + "\n")


def main():
    driver = webdriver.Chrome()
    driver.get("https://libraryguides.missouri.edu/c.php?g=1052498&p=7642280")
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@id='s-lg-side-nav-content']")
        )
    )
    # last one is content
    ATU_batches = driver.find_elements(By.PARTIAL_LINK_TEXT, "ATU ")[:-1]
    ATU_batches += driver.find_elements(By.LINK_TEXT, "ATU-1200- 2335")
    # store links instead of ele so that no stale ele ref
    ATU_batches = [ele.get_attribute("href") for ele in ATU_batches]
    for link in ATU_batches:
        driver.get(link)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, "//div[@id='s-lg-side-nav-content']")
            )
        )
        click_pitts_texts(driver)
    export_jsonl()


def debug(url):
    driver = webdriver.Chrome()
    driver.get(url)
    get_pitts_text(driver, debug=True)


if __name__ == '__main__':
    main()
    # debug("http://www.pitt.edu/~dash/type0001.html")
