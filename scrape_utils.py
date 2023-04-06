from bs4 import BeautifulSoup
import requests


# function created
def scrape(site):
    urls = []

    def scrape_helper(current_site):
        nonlocal urls
        # getting the request from url
        r = requests.get(current_site)
        # print(r.text)

        # converting the text
        s = BeautifulSoup(r.text, "html.parser")
        print(s.find_all("a"))
        for i in s.find_all("a"):
            # Check if 'href' exists in the attributes
            if "href" in i.attrs:
                href = i.attrs["href"]

                if href.startswith("/") or href.startswith("#"):
                    full_url = site + href

                    if full_url not in urls:
                        urls.append(full_url)
                        # calling it self
                        scrape_helper(full_url)

    scrape_helper(site)
    return urls
