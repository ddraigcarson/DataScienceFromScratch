import csv
import os.path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from card import Card


def transform_html_row_to_card(row):
    try:
        name = row[0].find('a', attrs={'class': 'card-link'}).text.strip()
        print("Creating card: " + name)
        character = row[1].text.strip()
        rarity = row[2].text.strip()
        type = row[3].text.strip()
        mana = sint(row[4].text.strip(), 0)
        card = Card(name, character, rarity, type, mana)

        card.effect = row[0].find('small').text.strip()
        card.attack = sint(row[5].text.strip(), 0)
        card.health = sint(row[6].text.strip(), 0)
        card.armour = sint(row[7].text.strip(), 0)
        return card
    except IndexError:
        print("Invalid row, returning row for cleaning")
        return row
    except ValueError:
        print("Invalid row, value error, return row for cleaning")
        return row


def sint(string, default):
    try:
        return int(string)
    except ValueError:
        return default


def get_card_list_from_net():
    rows = scrape_web_for_cards()
    cards = []
    cards_arr = []
    trash_rows = []
    for row in rows:
        cells = row.find_all('td')
        card = transform_html_row_to_card(cells)
        if type(card) is Card:
            cards.append(card)
            cards_arr.append(card.as_dict())
        else:
            trash_rows.append(card)

    df = pd.DataFrame.from_records(cards_arr)
    df.to_csv('cards.csv')

    with open('trash_rows.csv', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        for row in trash_rows:
            writer.writerow([row])
    return df


def scrape_web_for_cards():
    html = requests.get('http://www.hearthstonetopdecks.com/cards/page/1/?st=&manaCost=&format=standard&rarity=&type=&class=&set=&mechanic=&race=&orderby=ASC-name&view=table').text
    soup = BeautifulSoup(html, 'html5lib')

    page_limit_url = soup.find('a', attrs={'class': 'last'})['href']
    page_string = page_limit_url[page_limit_url.find('page'):][5:]
    page_limit = int(page_string[:page_string.find('/')])

    trs = []
    for page in range(1, page_limit+1):
        print("Getting page: " + str(page))
        url = 'http://www.hearthstonetopdecks.com/cards/page/{0}/?st=&manaCost=&format=standard&rarity=&type=&class=&set=&mechanic=&race=&orderby=ASC-name&view=table'.format(page)
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html5lib')
        card_list = soup.find('table', attrs={'id': 'card-list'})
        table_body = card_list.find('tbody')
        trs.extend(table_body.find_all('tr'))
    print("Got all pages")
    return trs


def plot_attack_vs_defense(df):
    print(df.columns.tolist())
    sns.lmplot(x='attack', y='health', data=df, fit_reg=False, hue='character')
    plt.ylim(0, None)
    plt.xlim(0, None)
    plt.show()


def plot_box_plot(df):
    print("Creating box plot")
    box_plot_df = df.drop(['i', 'armour'], axis=1)
    sns.boxplot(data=box_plot_df)
    plt.show()


def plot_swarm_plot(df):
    print("Creating swarm plot")
    stats_df = df.drop(['i', 'armour'], axis=1)
    melted_df = pd.melt(stats_df,
                        id_vars=['name', 'rarity', 'type', 'character', 'effect'],
                        var_name='stat')
    sns.swarmplot(x='stat', y='value', data=melted_df, hue='character')
    plt.show()


def run():
    df = None
    if os.path.isfile('cards.csv'):
        print("Cards already scraped; Loading from file")
        df = pd.read_csv('cards.csv')
    else:
        print("No local file; Getting from web")
        df = get_card_list_from_net()
    plot_attack_vs_defense(df)
    plot_box_plot(df)
    plot_swarm_plot(df)


run()
