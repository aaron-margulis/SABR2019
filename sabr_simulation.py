import math
import random
import pandas as pd
from datetime import datetime

LINEAR_WEIGHTS = {'k': -0.26, 'bb': 0.29, '1b': 0.44, '2b': 0.74, '3b': 1.01, 'hr': 1.39, 'hbp': 0.31, 'e': -0.26, 'oip': -0.26}

def sim_plate_app(hitter, pitcher):
	'''
	Simulate a plate appearence between hitter and pitcher using FanGraphs regressions

	Args:
		hitter (dictionary)
		pitcher (dictionary)

	Returns:
		simulated outcome: in the form 'k', 'bb', '1b', etc
		expected runs created
	'''
	# https://community.fangraphs.com/the-outcome-machine-predicting-at-bats-before-they-happen/
	k_pct = 2.71828**(.9427 * math.log(hitter['k_pct']) + .9254 * math.log(pitcher['k_pct']) + 1.5268)
	bb_pct = 2.71828**(.906 * math.log(hitter['bb_pct']) + .8644 * math.log(pitcher['bb_pct']) + 1.9975)
	single_pct = 2.71828**(1.01 * math.log(hitter['single_pct']) + 1.017 * math.log(pitcher['single_pct']) + 1.943)
	double_pct = .9206 * hitter['double_pct'] + .95779 * pitcher['double_pct'] - .03968
	triple_pct = 2.71828**(.8435 * math.log(hitter['triple_pct']) + .8698 * math.log(pitcher['triple_pct']) + 3.8809)
	hr_pct = 2.71828**(.9576 * math.log(hitter['hr_pct']) + .9268 * math.log(pitcher['hr_pct']) + 3.2129)
	hbp_pct = 2.71828**(.8761 * math.log(hitter['hbp_pct']) + .7623 * math.log(pitcher['hbp_pct']) + 2.995)
	e_pct = (1 - (k_pct + bb_pct + hbp_pct)) * .016 # errors occur on 1.6% of balls in play
	oip_pct = 1 - (k_pct + bb_pct + single_pct + double_pct + triple_pct + hr_pct + hbp_pct + e_pct) # out in play only remaining outcome

	exp_runs_created = LINEAR_WEIGHTS['bb'] * bb_pct + LINEAR_WEIGHTS['1b'] * single_pct + LINEAR_WEIGHTS['2b'] * double_pct + \
		LINEAR_WEIGHTS['3b'] * triple_pct + LINEAR_WEIGHTS['hr'] * hr_pct + LINEAR_WEIGHTS['hbp'] * hbp_pct + \
		LINEAR_WEIGHTS['k'] * k_pct + LINEAR_WEIGHTS['e'] * e_pct + LINEAR_WEIGHTS['oip'] * oip_pct

	probabilities = {'k': k_pct, 'bb': bb_pct, '1b': single_pct, '2b': double_pct, '3b': triple_pct, 'hr': hr_pct, 'hbp': hbp_pct,
		'e': e_pct, 'oip': oip_pct}
	simulated_outcome = random.choices(population=list(probabilities.keys()),
		weights=list(probabilities.values()), k=1)[0]

	return simulated_outcome, exp_runs_created


def sim_game(pitchers_df, hitters_df, using_opener):
	'''
	Simulate a game based on our assumptions about pitching changes

	Args:
		pitchers_df (dataframe)
		hitters_df (dataframe)
		using_opener (bool)

	Returns:
		sim_rc_per_hitter_per_inning: dataframe indicating expected runs created by each hitter in each inning
		sim_pa_per_hitter_per_inning: dataframe indicating which hitters batted in which innings 
	'''
	d = {k: [0]*9 for k in range(1,10)} # to create 9x9 df full of zeroes
	sim_rc_per_hitter_per_inning = pd.DataFrame(d) # runs created
	sim_pa_per_hitter_per_inning = pd.DataFrame(d) # plate appearances

	current_pitcher = 2 # starter
	if using_opener:
		current_pitcher = 1 # opener
	# current_pitcher = 3 corresponds to bullpen
	position_in_order = 0

	outs = 0
	total_plate_apps = 0
	outs_at_pitching_change = 0
	while outs < 27:
		total_plate_apps += 1
		position_in_order = position_in_order % 9 + 1

		inning = int(outs/3) + 1
		# used for visualization which shows probability of plate app in each inning
		# NOT expected number of plate apps in each inning.
		# index (representing innings) starts at 0, not 1
		sim_pa_per_hitter_per_inning.at[inning-1, position_in_order] = 1

		# when to change pitchers
		if using_opener:
			# change from opener to bridge after first 3 outs
			if outs == 3 and current_pitcher == 1:
				current_pitcher = 2
			# change from bridge to bullpen after 27 plate appearances
			if total_plate_apps > 27 and current_pitcher == 2:
				current_pitcher = 3
		else:
			# change from starter to 'opener' in relief after 24 plate appearances
			if total_plate_apps > 24 and current_pitcher == 2:
				current_pitcher = 1
				outs_at_pitching_change = outs
			# once 'opener' relief has entered, pitch 3 outs before going to bullpen
			if outs == outs_at_pitching_change + 3 and current_pitcher == 1:
				current_pitcher = 3
		
		hitter = hitters_df.iloc[position_in_order-1].to_dict() # simpler than using 'order' column
		pitcher = pitchers_df.loc[current_pitcher-1].to_dict()
		simulated_outcome, exp_runs_created = sim_plate_app(hitter, pitcher)
		if simulated_outcome in ['k', 'oip']:
			outs += 1

		# following two methods will converge at large enough sample size,
		# but using expected runs is more accurate
		# sim_rc_per_hitter_per_inning.loc[inning-1, position_in_order] += LINEAR_WEIGHTS[simulated_outcome]
		sim_rc_per_hitter_per_inning.loc[inning-1, position_in_order] += exp_runs_created

	return sim_rc_per_hitter_per_inning, sim_pa_per_hitter_per_inning


def simulation(pitchers_df, hitters_df, using_opener, sample_size):
	'''
	Simulate [sample_size] games and determine expected runs created based on pitching staff and order

	Args:
		pitchers_df (dataframe): 'order' column should follow: 1: opener, 2: starter, 3: bullpen
		hitters_df (dataframe)
		using_opener (bool)
		sample_size (int)

	Returns: average simulated runs created per game
	'''
	pitchers_df = pitchers_df.sort_values(by=['order']).reset_index(drop=True)
	hitters_df = hitters_df.sort_values(by=['order']).reset_index(drop=True)

	d = {k: [0]*9 for k in range(1,10)} # to create 9x9 df full of zeroes
	sum_sim_rc_per_hitter_per_inning = pd.DataFrame(d) # runs created
	sum_sim_pa_per_hitter_per_inning = pd.DataFrame(d) # plate appearances

	for i in range(sample_size):
		game_sim_rc_df, game_sim_pa_df = sim_game(pitchers_df, hitters_df, using_opener)
		sum_sim_rc_per_hitter_per_inning = sum_sim_rc_per_hitter_per_inning.add(game_sim_rc_df)
		sum_sim_pa_per_hitter_per_inning = sum_sim_pa_per_hitter_per_inning.add(game_sim_pa_df)

	sim_rc_df = sum_sim_rc_per_hitter_per_inning.div(sample_size)
	sim_pa_df = sum_sim_pa_per_hitter_per_inning.div(sample_size)
	# print(sim_pa_df) # used for data visualization to show likelihood of hitter getting plate app in each inning

	# sim_pa_df utilized for presentation visualization, no need to return
	return sim_rc_df.sum().sum() # total simulated runs created


def main():
	pitchers_df = pd.read_csv('pitcher_pcts.csv')
	hitters_df = pd.read_csv('hitter_pcts.csv')
	sample_size = 500 # expect runtime to be slightly over 0.01 seconds per game

	opener_sim_rc = simulation(pitchers_df, hitters_df, True, sample_size)
	starter_sim_rc = simulation(pitchers_df, hitters_df, False, sample_size)
	print('O: {}\nS: {}'.format(opener_sim_rc, starter_sim_rc))


if __name__ == '__main__':
	main()
