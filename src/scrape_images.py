from selenium.webdriver import Firefox
import time
import random
import os

from PIL import Image
import io

from image_manager import ImageManager

class ImageScraper():
    def __init__(self, image_manager):
        self.manager = image_manager

        # Create zombie browsers
        self.browser = Firefox()
        self.photo_browser = Firefox()

        #random.gammavariate(alpha=2, beta=2) #mean=alpha/beta

    def search_for_items(self, search_term):
        self.browser.get("https://images.google.com/")
        #time.sleep(2)
        time.sleep(random.gammavariate(alpha=2, beta=1))
        elem = self.browser.switch_to_active_element()
        search_box = elem
        search_box.click()
        #time.sleep(1)
        time.sleep(random.gammavariate(alpha=2, beta=2))
        search_box.send_keys(search_term)
        #time.sleep(1)
        time.sleep(random.gammavariate(alpha=2, beta=2))
        search_box.send_keys('\n')

    def find_images(self):
        """Return image objects"""
        time.sleep(5)
        #time.sleep(random.gammavariate(alpha=2.5, beta=0.5))
        images = self.browser.find_elements_by_css_selector(
            "div.rg_bx a.rg_l img")
        return images

    def scrape_image(self, image, directory, search_term, i):
        _ = image.location_once_scrolled_into_view
        image_url = image.get_attribute('src')
        self.photo_browser.get(image_url)
        image = self.photo_browser.find_element_by_css_selector(
            "img")
        image_png = image.screenshot_as_png
        #Check if we already have an image with the same hash value
        PIL_image = Image.open(io.BytesIO(image_png))
        if not self.image_manager.hash_exists(PIL_image):
            label = "_".join(search_term.lower().split())
            filepath = os.path.join(directory, 'image_{label}_{i}.png')
            # filepath = f'{directory}/image_{label}_{i}.png'
            with open(filepath, 'wb') as f:
                f.write(image_png)

    def scrape_images(self, images, directory, search_term, start_index=0):
        for i, image in enumerate(images):
            #time.sleep(1)
            time.sleep(random.gammavariate(alpha=2, beta=2))
            self.scrape_image(image, directory, search_term, i+start_index)

    def search_and_scrape(self, search_terms_for_directories, base_directory):
        for directory, search_terms in search_terms_for_directories.items():
            directory = os.path.join(base_directory, directory)
            for search_term in search_terms:
                self.search_for_items(search_term)
                n_images_scraped = 0
                while True:
                    images = self.find_images()
                    if len(images) <= n_images_scraped:
                        break
                    self.scrape_images(images[n_images_scraped:], directory, search_term,
                                  start_index=n_images_scraped)
                    n_images_scraped = len(images)

if __name__=='main':
    manager = ImageManager()

    search_terms_for_directories = {
        'betula_papyrifera': ['betula papyrifera']
    #     ,'acer_macrophylum': ['bigleaf maple forest', 'acer macrophylum tree', 'bigleaf maple branches']
    #     ,'platanus_acerifolia': ['london plane tree', 'london plane flowers']
    #     ,'pseudotsuga_menziesii': ['pseudotsuga menziesii tree', 'douglas fir needles']
    }

    base_directory = '~/tree-logic/tree_photos'

    search_and_scrape(search_terms_for_directories, base_directory)
