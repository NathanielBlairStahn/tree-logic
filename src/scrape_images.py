from selenium.webdriver import Firefox
import time
import random
import os

import PIL
import io

from image_manager import ImageManager

class ImageScraper():
    def __init__(self, image_manager, random_sleep=True):
        self.manager = image_manager

        # Create zombie browsers
        self.browser = Firefox()
        self.photo_browser = Firefox()

        #random.gammavariate(alpha=2, beta=2) #mean=alpha/beta
        self.timer = random.gammavariate if random_sleep else None
        self.alpha = 2 #alpha parameter for the gamma random variable

    def get_time(self, mean_time):
        if self.timer is not None:
            return self.timer(alpha=self.alpha, beta=self.alpha / mean_time)
        else:
            return mean_time

    def search_for_items(self, search_term):
        self.browser.get("https://images.google.com/")
        #time.sleep(2)
        time.sleep(get_time(2))
        elem = self.browser.switch_to_active_element()
        search_box = elem
        search_box.click()
        #time.sleep(1)
        time.sleep(get_time(1))
        search_box.send_keys(search_term)
        #time.sleep(1)
        time.sleep(get_time(1))
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
        #image will be an object of type
        #<class selenium.webdriver.remote.webelement.WebElement>
        image = self.photo_browser.find_element_by_css_selector(
            "img")
        #The screenshot_as_png @property of the WebElement object
        #returns a screenshot of the element as binary data of type
        #<class bytes>
        image_png = image.screenshot_as_png
        #To get a PIL object that we can pass to the image manager
        #and compute a hash value from we need to open the bytes by
        #first converting them using BytesIO.
        PIL_image = PIL.Image.open(io.BytesIO(image_png))
        #Check if we have already found an image with the same hash value
        if not self.image_manager.knows_image(PIL_image):
            label = "_".join(search_term.lower().split())
            filepath = os.path.join(directory, 'image_{label}_{i}.png')
            # filepath = f'{directory}/image_{label}_{i}.png'
            with open(filepath, 'wb') as f:
                f.write(image_png)
            #Presumably this would also work if we wrote the PIL object...

    def scrape_images(self, images, directory, search_term, start_index=0):
        for i, image in enumerate(images):
            #time.sleep(1)
            time.sleep(get_time(1))
            self.scrape_image(image, directory, search_term, i+start_index)

    def search_and_scrape(self, search_terms_for_directories, base_directory):
        for directory, search_terms in search_terms_for_directories.items():
            directory = os.path.join(base_directory, directory)
            if not os.path.exists(directory):
                os.makedirs(directory)
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
