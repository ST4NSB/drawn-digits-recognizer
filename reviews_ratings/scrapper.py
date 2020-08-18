import subprocess

from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import os
import pathlib
import random

processed = 0

def simple_get(url):
    try:
        headers = {"Accept-Language": "en-US, en;q=0.5"}
        with closing(get(url, stream=True, headers=headers)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    print(e)


def create_dir(path):
    # if already exists it will fail creation
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else: print("Successfully created the directory %s" % path)

    try:
        os.mkdir(path + "\\train")
    except OSError:
        print("Creation of the directory %s failed" % path + "\\train")
    else: print("Successfully created the directory %s" % path + "\\train")

    try:
        os.mkdir(path + "\\test")
    except OSError:
        print("Creation of the directory %s failed" % path + "\\test")
    else: print("Successfully created the directory %s" % path + "\\test")

    type = ["train", "test"]
    ratings = ["bad", "good"]
    for i in type:
        for j in ratings:
            try:
                dir = path + "\\" + str(i) + "\\" + str(j)
                os.mkdir(dir)
            except OSError:
                print("Creation of the directory %s failed" % dir)
            else:
                print("Successfully created the directory %s" % dir)


def get_movie_links(url):

    raw_html = simple_get(url)
    html = BeautifulSoup(raw_html, "html.parser")

    movies = html.find("ul", {"class": "listare_filme"}).find_all("li")
    links = set()
    for movie in movies:
        link = movie.find("h2").find("a", href=True)['href']
        links.add(link)
    return links

def convert_grade(grade):
    grade = int(grade)
    if grade <= 4:
        return "bad"
    elif grade >= 7 and grade <= 10:
        return "good"
    return "none"

def convert_romanian_diacritics(text):
    new_text = text
    new_text = new_text.replace("ă", "a")
    new_text = new_text.replace("â", "a")
    new_text = new_text.replace("î", "i")
    new_text = new_text.replace("ș", "s")
    new_text = new_text.replace("ț", "t")

    new_text = new_text.replace("Ă", "A")
    new_text = new_text.replace("Â", "A")
    new_text = new_text.replace("Î", "I")
    new_text = new_text.replace("Ș", "S")
    new_text = new_text.replace("Ț", "T")
    return new_text

def clear():
    print(' \n' * 25)

def console(max_cap):
    #os.system('cls' if os.name == 'nt' else 'clear')
    clear()
    print("DONE: {} / {}".format(processed, max_cap * 2))

def create_movie_review(path, movie_name, page, labels, max_cap):
    raw_html = simple_get(page)  # top rated movies
    html = BeautifulSoup(raw_html, 'html.parser')

    div = html.find("div", {"class": "box_movie_comment"}).find("div", {"id": "user_reviews"})

    if div is None:
        return

    div = div.find_all("div", {"class": "post clearfix"})

    iter = 0
    for review in div:

        if labels.get("bad") >= max_cap and labels.get("good") >= max_cap:
            return

        grade = review.find("span", {"class": "stelutze"})
        if grade is None:
            continue

        grade = grade.find_all("img", src=True)
        final_grade = 0
        for img in grade:
            if img['src'].find('star_full') != -1:
                final_grade += 1

        new_grade = convert_grade(final_grade)

        if new_grade == "none":
            continue

        if labels.get(new_grade) >= max_cap:
            continue
        else:
            labels[new_grade] += 1

        text = review.find("div", {"class": "left comentariu"}).find("div", {"class": "mb5"}).find("span").text
        text = convert_romanian_diacritics(text)
        #print("Old grade: {}  |  New grade: {}  |  Text: {}".format(final_grade, new_grade, text))

        ftxt = path
        if labels.get(new_grade) > (max_cap / 2):
            ftxt += "\\test\\"
        else:
            ftxt += "\\train\\"
        ftxt += new_grade + "\\" + movie_name + "_" + str(iter + 1) + ".txt"
        with open(ftxt, "w") as file:
            file.write(str(text.encode('utf8')))

        global processed
        processed += 1
        console(max_cap)
        iter += 1


if __name__ == "__main__":

    path = str(pathlib.Path().absolute()) + "\\cinemagia_reviews"
    create_dir(path)

    urls = ["https://www.cinemagia.ro/filme-poster/?layout=3&pn=", "https://www.cinemagia.ro/seriale-tv-poster/?layout=3&pn="]
    links = set()
    page_nr = 54
    for url in urls:
        for page in range(1, page_nr + 1):
            link = url + str(page)
            links.update(get_movie_links(link))

    links = list(links)
    random.shuffle(links)

    max_cap = 3500 # 3700
    labels = {
        "bad": 0,
        "good": 0
    }

    print(len(links))
    print(len(set(links)))

    iter = 0
    for movie in links:
        #print("Link: {}".format(movie))
        if labels.get("bad") >= max_cap and labels.get("good") >= max_cap:
            break
        create_movie_review(path, "m_" + str(iter + 1), movie, labels, max_cap)
        iter += 1