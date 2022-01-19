# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 00:11:12 2020


CFB_RISK.PY
@author: Connor
"""

import requests as reqs

import random
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import date
plt.style.use("bmh")

def pca():
    """
    please close all
    """
    plt.close("all")

def yline(loc, *args, ax=None, **kwargs):
    if ax is None:
        ylims = plt.ylim()
        plt.plot([loc, loc], ylims, *args, **kwargs)
        plt.ylim(ylims)
    else:
        ylims = ax.get_ylim()
        ax.plot([loc, loc], ylims, *args, **kwargs)
        ax.set_ylim(ylims)
#%%
# SAVE_FLAG = True
# REPLACE_FLAG = True

# if SAVE_FLAG:
#     figs_base_dir = Path("D:/Connor/Documents/AZ 2020/cfb_risk")
#     check_dir = figs_base_dir / f"{date.today().isoformat()}"
#     # check_dir = figs_base_dir / "2020-04-22"
#     asserted_dir = figs_base_dir / "temp_dir"
#     # asserted_dir = check_dir
#     if not check_dir.exists():
#         os.mkdir(check_dir)
#         save_dir = check_dir
#     else:
#         if REPLACE_FLAG:
#             save_dir = check_dir
#         else:
#             save_dir = asserted_dir

# base = "https://collegefootballrisk.com/api"

# # get tech players
# players_req = reqs.get(base+"/players", params={"team": "Georgia Tech"})
# gt_players_info = players_req.json()

# # get tech team stats
# stats_req = reqs.get(base+"/stats/team", params={"team": "Georgia Tech"})
# gt_info = stats_req.json()

# l1 = [player for player in gt_players_info if player["lastTurn"]["season"]]
# s2_players = [player for player in l1 if player["lastTurn"]["season"]>1]
# max_day = max([player["lastTurn"]["day"] for player in s2_players])

# team_odds_req = reqs.get(base+"/team/odds",
#                          params={"season": 2,
#                                  "day": max_day,
#                                  "team": "Georgia Tech"})
# team_odds_info = team_odds_req.json()
# #% Make Territory Plots for GT
# # territory_req = reqs.get(base+"/territories",
# #                           params={"season": 2,
# #                                   "day": max_day})
# # territory_list = territory_req.json()

# gt_territories = [terry["territory"] for terry in team_odds_info[:-1]]

# pca()
# plot_num = 1
# lowest_win = 1
# win_terr = None
# highest_loss = 1
# loss_terr = None
# for terry in gt_territories:
#     territory_info = reqs.get(base+"/territory/turn",
#                               params={"territory": terry,
#                                       "season": 2,
#                                       "day": max_day})
#     terry_json = territory_info.json()
#     teams = terry_json["teams"]
#     # set up powers
#     powers = [team["power"] for team in teams]
#     # set up labels
#     labels = [team["team"] for team in teams]
#     # set up colors
#     colors = [team["color"] for team in teams]
#     # set up "explode"
#     explode = [0.02]*len(teams)
#     for ind, team in enumerate(teams):
#         if team["team"] == terry_json["winner"]:
#             # explode[ind] = 0.02
#             num_spaces = 0
#             if len(team["team"]) > len("(winner)"):
#                 num_spaces = (len(team["team"]) - len("(winner)"))//2
#             labels[ind] += "\n(winner)"+" "*num_spaces
            
#             winner_chance = terry_json["teams"][ind]["chance"]
#             if team["team"] == "Georgia Tech":
#                 # GT won, huzzah!
#                 if winner_chance < lowest_win:
#                     lowest_win = winner_chance
#                     win_terr = terry
#                     low_pie = {
#                         "terry": terry,
#                         "powers": powers,
#                         "explode": explode,
#                         "labels": labels,
#                         "colors": colors,
#                         "autopct": "%2.2f%%",
#                         "pctdistance": 0.4,
#                         "startangle": 90}
#             else:
#                 if winner_chance < highest_loss:
#                     highest_loss = winner_chance
#                     loss_terr = terry
#                     hi_pie = {
#                         "terry": terry,
#                         "powers": powers,
#                         "explode": explode,
#                         "labels": labels,
#                         "colors": colors,
#                         "autopct": "%2.2f%%",
#                         "pctdistance": 0.4,
#                         "startangle": 90}
                
#     if len(powers) > 1:
#         plt.figure(plot_num)
#         ax = plt.gca()
#         plt.title(terry)
#         patches, texts, autotexts = plt.pie(
#             powers,             # num_players for each school in territory
#             explode=explode,    # pops out winning team
#             labels=labels,      # Sets the labels for each school in territory
#             colors=colors,      # Sets the colors to the school for each territory
#             autopct="%2.2f%%",  # Sets the percent formating
#             pctdistance=0.4,
#             startangle=90       # Sets angle to start pie pieces at
#             )
#         centre_circle = plt.Circle((0,0),0.70,fc='white')
#         ax.add_artist(centre_circle)
        
#         if SAVE_FLAG:
#             plt.savefig(save_dir / f"{terry}.png", dpi=150)
#         plt.close()
#         plot_num+=1

#%% Make Expected Value Histogram
# plt.figure(plot_num)
# odds = np.ones((len(team_odds_info)-1,))
# from scipy import stats

# for ind, tory in enumerate(team_odds_info[:-1]):
#     if tory["territoryPower"]>0:
#         odds[ind] = tory["teamPower"] / tory["territoryPower"]

# num_runs = 100000
# run_vec = np.ones((num_runs,))

# for run in range(num_runs):
#     out_vals = np.ones((len(odds),))
#     r_vals = np.random.uniform(size=len(odds))
    
#     for ind, terry in enumerate(odds):
#         if terry < r_vals[ind]:
#             out_vals[ind] = 0 # set it to a loss
#             # else we won, leave as a win
#     # after run loop is done, see how many we won by summing
#     run_vec[run] = sum(out_vals)

# #%
# vals, bins, patchs = plt.hist(run_vec, color="#4B8B9B", density=True, bins=np.arange(0,len(odds)))
# exp = sum(odds)
# act = gt_info["territories"]
# mu = np.mean(run_vec)
# sigma = np.std(run_vec)
# dsigma = (act-mu) / sigma # get dSigma
# fit_mu, fit_sigma = stats.norm.fit(run_vec)
# x = np.linspace(0, len(odds), 5000)

# y = (1 / (np.sqrt(2 * np.pi * np.power(fit_sigma, 2)))) * \
#     (np.power(np.e, -(np.power((x - fit_mu), 2) / (2 * np.power(fit_sigma, 2)))))
# plt.plot(x,y, linestyle="-", linewidth=0.5, color="#54585A", label="Fit PDF")
# draw_percentage = stats.norm.pdf(dsigma)
# yline(exp, linestyle=(0,(2,2)), linewidth=2, color="#003057", label="Expected Value")
# yline(act, linestyle=(0,(2,2)), linewidth=2, color="#EAAA00", label="Actual Territories")
# plt.title(f"Number of Territories Histogram\nExpected: {exp:2.2f}        Actual: {act}")
# plt.xlabel("Number of Territories Won")
# plt.ylabel("Percent Chance to Win N Territories (%)")
# my_anno_text = f"""$\mu = {mu:2.3f}$
# $3\sigma = {3*sigma:2.3f}$
# $\Delta\sigma = {dsigma:2.3f}$
# $P(Draw) = {100*draw_percentage:2.3f}\%$"""
# ax = plt.gca()
# ax.text(0,ax.get_ylim()[1]*0.80, my_anno_text, bbox={'facecolor': 'white', 'alpha': 0.8})
# ax.legend(loc="lower left")
# if SAVE_FLAG:
#     plt.savefig(save_dir / f"num_of_territories_hist.png", dpi=150)
# # plt.close()
# plot_num+=1

# print("lowest win was", lowest_win, "in", win_terr)
# print("highest loss was", highest_loss, "in", loss_terr)
# #%
# fig, axes = plt.subplots(1,3, figsize=(18,7))
# for ind, ax in enumerate(axes):
#     if ind == 0:
#         terry = low_pie["terry"]
#         powers = low_pie["powers"]
#         explode = low_pie["explode"]
#         labels = low_pie["labels"]
#         colors = low_pie["colors"]
#         ax.set_title(terry)
#         patches, texts, autotexts = ax.pie(
#             powers,             # num_players for each school in territory
#             explode=explode,    # pops out winning team
#             labels=labels,      # Sets the labels for each school in territory
#             colors=colors,      # Sets the colors to the school for each territory
#             autopct="%2.2f%%",  # Sets the percent formating
#             pctdistance=0.4,
#             startangle=90       # Sets angle to start pie pieces at
#             )
#         centre_circle = plt.Circle((0,0),0.70,fc='white')
#         ax.add_artist(centre_circle)
#     if ind == 1:
#         ax.hist(run_vec, color="#4B8B9B", density=True, bins=np.arange(0,len(odds)))        
#         exp = sum(odds)
#         act = gt_info["territories"]
#         yline(exp, linestyle=(0,(2,2)), linewidth=2, color="#003057", ax=ax, label="Expected Value")
#         yline(act, linestyle=(0,(2,2)), linewidth=2, color="#EAAA00", ax=ax, label="Actual Territories")
#         ax.set_title(f"Number of Territories Histogram\nExpected: {exp:2.2f}         Actual: {act}")
#         ax.set_xlabel("Number of Territories Won")
#         ax.set_ylabel("Chance to Win N Territories")
#         ax.text(0,ax.get_ylim()[1]*0.85, my_anno_text, bbox={'facecolor': 'white', 'alpha': 0.8})
#         ax.legend(loc="lower left")
#     if ind == 2:
#         terry = hi_pie["terry"]
#         powers = hi_pie["powers"]
#         explode = hi_pie["explode"]
#         labels = hi_pie["labels"]
#         colors = hi_pie["colors"]
#         ax.set_title(terry)
#         patches, texts, autotexts = ax.pie(
#             powers,             # num_players for each school in territory
#             explode=explode,    # pops out winning team
#             labels=labels,      # Sets the labels for each school in territory
#             colors=colors,      # Sets the colors to the school for each territory
#             autopct="%2.2f%%",  # Sets the percent formating
#             pctdistance=0.4,
#             startangle=90       # Sets angle to start pie pieces at
#             )
#         centre_circle = plt.Circle((0,0),0.70,fc='white')
#         ax.add_artist(centre_circle)

# if SAVE_FLAG:
#     fig.savefig(save_dir / "win_exphist_loss.png", dpi=150)

#%%
# TODO
import numpy as np
import multiprocessing
from multiprocessing import Pool, freeze_support

if __name__ == "__main__":
    freeze_support()

odds = [0.25, 0.25, 0.3, 0.3, 0.4, 0.8, 0.9, 1, 1]

def runs_mapper(func, num_runs, num_workers, odds):
    if num_runs % num_workers:
        arr = np.array([num_runs // num_workers]*num_workers)
        for i in range(num_runs % num_workers):
            arr[i] += 1
    else:
        arr = np.array([num_runs // num_workers]*num_workers)
    
    print("arr = ", arr)
    print("odds = ", odds)
    
    arr_of_tuples = [(arr_i, odds) for arr_i in arr]
    print(arr_of_tuples)
    
    with Pool(num_workers) as p:
        res = p.map(func, arr_of_tuples)
    
    return res

def runs_runner(values):
    num_runs, odds = values
    print(num_runs, odds)
    num_territories = len(odds)
    run_vec = np.ones((num_runs,))
    for run in range(num_runs):
        out_vals = np.ones((num_territories,))
        r_vals = np.random.uniform(size=num_territories)
        
        for ind, terry in enumerate(odds):
            if terry < r_vals[ind]:
                out_vals[ind] = 0
        
        run_vec[run] = sum(out_vals)
    
    return run_vec

runs_mapper(runs_runner, 1000, 4, odds)

# from workers import f

# if __name__ == "__main__":
#     p = Pool(3)
#     for n in p.map(f, [1,2,3]):
#         print(n)

# run_vec = np.ones((num_runs,))
# for run in range(num_runs):
#     out_vals = np.ones((num_territories,))
#     r_vals = np.random.uniform(size=num_territories)
    
#     for ind, terry in enumerate(odds):
#         if terry < r_vals[ind]:
#             out_vals[ind] = 0
            
#     run_vec[run] = sum(out_vals)