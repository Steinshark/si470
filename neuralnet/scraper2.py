#pip3 install selenium Pillow

#Adapted from https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d

import selenium
from selenium import webdriver
import time,os
from PIL import Image
import io
import requests
import hashlib
import argparse
import random
from glob import glob

DRIVER_PATH="./chromedriver"
target_path='./images'

parser = argparse.ArgumentParser()
parser.add_argument('search_term')
args=parser.parse_args()

def fetch_image_urls(query, max_links_to_fetch, wd, sleep_between_interactions=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))
            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            return
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls

def persist_image(folder_path,url):
    try:
        image_content = requests.get(url,verify=False).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")

def search_and_download(search_term,driver_path,target_path='./images',number_images=100):
    target_folder = os.path.join(target_path,'_'.join(search_term.lower().split(' ')))
    if not os.path.exists(target_folder):
      os.makedirs(target_folder)

    with webdriver.Chrome(executable_path=driver_path) as wd:
        res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0.5)
        print(f"res: {res}")
    for elem in res:
      persist_image(target_folder,elem)

search_and_download(search_term=args.search_term,driver_path=DRIVER_PATH,number_images=120,target_path=target_path)
os.chdir(target_path)
for dataset in ['training','validation']:
  tgt=f'{dataset}/{args.search_term}'
  if not os.path.exists(tgt):
    os.makedirs(tgt)
imgs=list(glob(f'{args.search_term}/*'))
random.shuffle(imgs)
for img in imgs[:int(len(imgs)/6)]:
  fn=os.path.basename(img)
  os.rename(img,f'validation/{args.search_term}/{fn}')
for img in imgs[int(len(imgs)/6):]:
  fn=os.path.basename(img)
  os.rename(img,f'training/{args.search_term}/{fn}')
os.rmdir(args.search_term)