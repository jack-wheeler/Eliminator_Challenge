import numpy as np

def nfl_dp(prob_matrix):
    n, m = prob_matrix.shape
    median_win_prob = np.median(prob_matrix, axis=1)
    team_indices_to_remove = np.argsort(median_win_prob)[:12]
    prob_matrix = np.delete(prob_matrix, team_indices_to_remove, axis=0)
    n, m = prob_matrix.shape
    dp = np.zeros((2**n, m))
    teams_picked = np.zeros((2**n, m), dtype=np.int32)
    dp[:,0] = 1
    for j in range(1,m):
        for i in range(2**n):
            for k in range(n):
                if (i & (1 << k)) != 0:
                    if dp[i^(1 << k), j-1] * prob_matrix[k,j] > dp[i,j]:
                        dp[i,j] = dp[i^(1 << k), j-1] * prob_matrix[k,j]
                        teams_picked[i,j] = k
    return dp, teams_picked


def get_optimal_picks(dp, teams_picked, team_dict):
    n, m = dp.shape
    team_names = []
    i = np.argmax(dp[:,-1])
    for j in range(m-1, -1, -1):
        team_index = np.unravel_index(teams_picked[i, j], (n,))[0]
        team_names.append(team_dict[team_index])
        i ^= 1 << team_index
    return list(reversed(team_names))


if __name__ == '__main__':
    team_dict = {'New England Patriots': 0, 'Kansas City Chiefs': 1, 'Houston Texans': 2, 'Philadelphia Eagles': 3, 'Chicago Bears': 4, 'New Orleans Saints': 5, 'Los Angeles Rams': 6, 'Pittsburgh Steelers': 7, 'Indianapolis Colts': 8, 'Cleveland Browns': 9, 'Green Bay Packers': 10, 'San Francisco 49ers': 11, 'Denver Broncos': 12, 'Minnesota Vikings': 13, 'Seattle Seahawks': 14, 'Baltimore Ravens': 15, 'Dallas Cowboys': 16, 'Los Angeles Chargers': 17, 'Tennessee Titans': 18, 'New York Giants': 19}

    prob_matrix = np.random.rand(32, 18)
    dp, team_picked = nfl_dp(prob_matrix)
    print(get_optimal_picks(dp,team_picked,team_dict))
