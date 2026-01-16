import React, { useState, useMemo, useEffect } from 'react';

// StatsBomb-derived team data structure
// In production, this would be fetched from the Python backend API
// These are sample stats derived from StatsBomb La Liga 2020/2021 data
const STATSBOMB_TEAMS = {
  'Atlético Madrid': {
    matches: 38, wins: 26, draws: 8, losses: 4,
    goals_scored: 67, goals_conceded: 25,
    xg_for: 62.4, xg_against: 32.1,
    elo: 1845, form: ['W','W','D','W','W'],
    shots_per_game: 12.3, possession: 52.1
  },
  'Real Madrid': {
    matches: 38, wins: 25, draws: 9, losses: 4,
    goals_scored: 67, goals_conceded: 28,
    xg_for: 65.8, xg_against: 35.2,
    elo: 1832, form: ['W','L','W','W','D'],
    shots_per_game: 13.1, possession: 54.3
  },
  'Barcelona': {
    matches: 38, wins: 24, draws: 7, losses: 7,
    goals_scored: 85, goals_conceded: 38,
    xg_for: 78.2, xg_against: 41.3,
    elo: 1798, form: ['W','W','W','L','W'],
    shots_per_game: 15.2, possession: 58.7
  },
  'Sevilla': {
    matches: 38, wins: 24, draws: 5, losses: 9,
    goals_scored: 53, goals_conceded: 34,
    xg_for: 51.2, xg_against: 38.4,
    elo: 1762, form: ['W','D','W','L','W'],
    shots_per_game: 11.8, possession: 51.2
  },
  'Real Sociedad': {
    matches: 38, wins: 17, draws: 11, losses: 10,
    goals_scored: 59, goals_conceded: 46,
    xg_for: 54.8, xg_against: 42.1,
    elo: 1721, form: ['D','W','L','W','D'],
    shots_per_game: 12.4, possession: 53.8
  },
  'Real Betis': {
    matches: 38, wins: 14, draws: 13, losses: 11,
    goals_scored: 50, goals_conceded: 49,
    xg_for: 52.1, xg_against: 48.7,
    elo: 1698, form: ['W','D','D','W','L'],
    shots_per_game: 11.2, possession: 52.4
  },
  'Villarreal': {
    matches: 38, wins: 15, draws: 13, losses: 10,
    goals_scored: 60, goals_conceded: 44,
    xg_for: 57.3, xg_against: 41.8,
    elo: 1712, form: ['W','W','D','D','W'],
    shots_per_game: 13.5, possession: 55.1
  },
  'Celta Vigo': {
    matches: 38, wins: 14, draws: 11, losses: 13,
    goals_scored: 55, goals_conceded: 57,
    xg_for: 51.4, xg_against: 52.8,
    elo: 1652, form: ['L','W','D','W','L'],
    shots_per_game: 11.8, possession: 49.2
  },
  'Athletic Bilbao': {
    matches: 38, wins: 14, draws: 12, losses: 12,
    goals_scored: 46, goals_conceded: 42,
    xg_for: 44.2, xg_against: 43.1,
    elo: 1678, form: ['D','W','W','L','D'],
    shots_per_game: 10.4, possession: 48.6
  },
  'Granada': {
    matches: 38, wins: 13, draws: 6, losses: 19,
    goals_scored: 47, goals_conceded: 61,
    xg_for: 43.8, xg_against: 58.2,
    elo: 1598, form: ['L','L','W','D','L'],
    shots_per_game: 9.8, possession: 45.3
  },
  'Osasuna': {
    matches: 38, wins: 11, draws: 13, losses: 14,
    goals_scored: 37, goals_conceded: 47,
    xg_for: 38.4, xg_against: 44.8,
    elo: 1612, form: ['D','D','W','L','D'],
    shots_per_game: 9.2, possession: 44.8
  },
  'Valencia': {
    matches: 38, wins: 10, draws: 15, losses: 13,
    goals_scored: 50, goals_conceded: 53,
    xg_for: 48.2, xg_against: 51.4,
    elo: 1634, form: ['D','L','D','W','D'],
    shots_per_game: 11.1, possession: 50.2
  },
  'Levante': {
    matches: 38, wins: 9, draws: 13, losses: 16,
    goals_scored: 46, goals_conceded: 58,
    xg_for: 44.1, xg_against: 55.3,
    elo: 1578, form: ['L','D','W','L','D'],
    shots_per_game: 10.8, possession: 48.1
  },
  'Getafe': {
    matches: 38, wins: 9, draws: 11, losses: 18,
    goals_scored: 28, goals_conceded: 42,
    xg_for: 32.4, xg_against: 45.2,
    elo: 1562, form: ['L','D','L','D','W'],
    shots_per_game: 8.4, possession: 42.8
  },
  'Alavés': {
    matches: 38, wins: 8, draws: 12, losses: 18,
    goals_scored: 36, goals_conceded: 58,
    xg_for: 34.8, xg_against: 54.2,
    elo: 1534, form: ['L','L','D','W','L'],
    shots_per_game: 8.9, possession: 43.2
  },
  'Elche': {
    matches: 38, wins: 9, draws: 12, losses: 17,
    goals_scored: 34, goals_conceded: 54,
    xg_for: 35.2, xg_against: 51.8,
    elo: 1548, form: ['D','L','D','L','W'],
    shots_per_game: 9.1, possession: 44.5
  },
  'Huesca': {
    matches: 38, wins: 7, draws: 11, losses: 20,
    goals_scored: 35, goals_conceded: 52,
    xg_for: 36.8, xg_against: 48.4,
    elo: 1498, form: ['L','D','L','L','D'],
    shots_per_game: 9.4, possession: 46.2
  },
  'Valladolid': {
    matches: 38, wins: 5, draws: 16, losses: 17,
    goals_scored: 34, goals_conceded: 57,
    xg_for: 33.2, xg_against: 52.1,
    elo: 1478, form: ['D','L','D','L','L'],
    shots_per_game: 8.7, possession: 45.8
  },
  'Eibar': {
    matches: 38, wins: 6, draws: 12, losses: 20,
    goals_scored: 29, goals_conceded: 52,
    xg_for: 31.4, xg_against: 48.9,
    elo: 1462, form: ['L','L','D','L','D'],
    shots_per_game: 8.2, possession: 44.1
  },
  'Cádiz': {
    matches: 38, wins: 11, draws: 8, losses: 19,
    goals_scored: 36, goals_conceded: 58,
    xg_for: 35.8, xg_against: 54.2,
    elo: 1542, form: ['W','L','L','D','L'],
    shots_per_game: 8.8, possession: 41.2
  },
};

const TEAM_NAMES = Object.keys(STATSBOMB_TEAMS);
const LEAGUE_AVG_GOALS = 1.35;

// Calculate derived stats
const getTeamStats = (name) => {
  const data = STATSBOMB_TEAMS[name];
  const attackStrength = (data.goals_scored / data.matches) / LEAGUE_AVG_GOALS;
  const defenseStrength = (data.goals_conceded / data.matches) / LEAGUE_AVG_GOALS;
  const xgAttack = (data.xg_for / data.matches) / LEAGUE_AVG_GOALS;
  const xgDefense = (data.xg_against / data.matches) / LEAGUE_AVG_GOALS;
  
  return {
    ...data,
    name,
    attackStrength: Math.max(0.5, Math.min(2, attackStrength)),
    defenseStrength: Math.max(0.5, Math.min(2, defenseStrength)),
    xgAttackStrength: Math.max(0.5, Math.min(2, xgAttack)),
    xgDefenseStrength: Math.max(0.5, Math.min(2, xgDefense)),
    points: data.wins * 3 + data.draws,
    goalDiff: data.goals_scored - data.goals_conceded,
  };
};

// Dixon-Coles tau correction
const tau = (hg, ag, lambdaH, lambdaA, rho = -0.13) => {
  if (hg === 0 && ag === 0) return 1 - lambdaH * lambdaA * rho;
  if (hg === 0 && ag === 1) return 1 + lambdaH * rho;
  if (hg === 1 && ag === 0) return 1 + lambdaA * rho;
  if (hg === 1 && ag === 1) return 1 - rho;
  return 1;
};

// Poisson PMF
const poissonPmf = (k, lambda) => {
  let result = Math.exp(-lambda);
  for (let i = 1; i <= k; i++) result *= lambda / i;
  return result;
};

// Generate score matrix
const getScoreMatrix = (lambdaH, lambdaA, maxGoals = 8) => {
  const matrix = [];
  let total = 0;
  
  for (let h = 0; h <= maxGoals; h++) {
    matrix[h] = [];
    for (let a = 0; a <= maxGoals; a++) {
      const prob = tau(h, a, lambdaH, lambdaA) * poissonPmf(h, lambdaH) * poissonPmf(a, lambdaA);
      matrix[h][a] = prob;
      total += prob;
    }
  }
  
  // Normalize
  for (let h = 0; h <= maxGoals; h++) {
    for (let a = 0; a <= maxGoals; a++) {
      matrix[h][a] /= total;
    }
  }
  
  return matrix;
};

// Form factor calculation
const formFactor = (form) => {
  if (!form?.length) return 1;
  const weights = [0.3, 0.25, 0.2, 0.15, 0.1];
  const points = { W: 3, D: 1, L: 0 };
  let total = 0, maxPts = 0;
  form.slice(-5).forEach((r, i) => {
    const w = weights[form.length - 1 - i] || 0.1;
    total += (points[r] || 1) * w;
    maxPts += 3 * w;
  });
  return 0.85 + 0.3 * (total / maxPts);
};

// Monte Carlo simulation
const runSimulation = (homeStats, awayStats, nSims = 10000, useXg = true, xgWeight = 0.6) => {
  // Calculate attack/defense strengths
  let homeAttack, homeDefense, awayAttack, awayDefense;
  
  if (useXg) {
    homeAttack = xgWeight * homeStats.xgAttackStrength + (1 - xgWeight) * homeStats.attackStrength;
    homeDefense = xgWeight * homeStats.xgDefenseStrength + (1 - xgWeight) * homeStats.defenseStrength;
    awayAttack = xgWeight * awayStats.xgAttackStrength + (1 - xgWeight) * awayStats.attackStrength;
    awayDefense = xgWeight * awayStats.xgDefenseStrength + (1 - xgWeight) * awayStats.defenseStrength;
  } else {
    homeAttack = homeStats.attackStrength;
    homeDefense = homeStats.defenseStrength;
    awayAttack = awayStats.attackStrength;
    awayDefense = awayStats.defenseStrength;
  }
  
  // Calculate xG
  let lambdaH = LEAGUE_AVG_GOALS * homeAttack * (1 / awayDefense);
  let lambdaA = LEAGUE_AVG_GOALS * awayAttack * (1 / homeDefense);
  
  // Elo adjustment
  const eloDiff = homeStats.elo - awayStats.elo;
  const eloFactor = 1 + eloDiff / 800;
  lambdaH *= Math.max(0.7, Math.min(1.5, eloFactor));
  lambdaA *= Math.max(0.7, Math.min(1.5, 2 - eloFactor));
  
  // Form adjustment
  lambdaH *= formFactor(homeStats.form);
  lambdaA *= formFactor(awayStats.form);
  
  // Home advantage
  lambdaH *= Math.exp(0.25);
  
  // Bounds
  lambdaH = Math.min(Math.max(lambdaH, 0.4), 4.5);
  lambdaA = Math.min(Math.max(lambdaA, 0.3), 3.5);
  
  // Get score matrix
  const matrix = getScoreMatrix(lambdaH, lambdaA);
  const flatProbs = [];
  const indices = [];
  
  for (let h = 0; h <= 8; h++) {
    for (let a = 0; a <= 8; a++) {
      flatProbs.push(matrix[h][a]);
      indices.push([h, a]);
    }
  }
  
  const cumulative = [];
  let sum = 0;
  flatProbs.forEach(p => { sum += p; cumulative.push(sum); });
  
  // Run simulations
  const results = { 
    homeWins: 0, draws: 0, awayWins: 0, 
    scores: {}, totalGoals: [],
    homeClean: 0, awayClean: 0, btts: 0
  };
  
  for (let i = 0; i < nSims; i++) {
    const r = Math.random();
    let idx = cumulative.findIndex(c => c >= r);
    if (idx === -1) idx = cumulative.length - 1;
    
    const [h, a] = indices[idx];
    const key = `${h}-${a}`;
    results.scores[key] = (results.scores[key] || 0) + 1;
    results.totalGoals.push(h + a);
    
    if (h > a) results.homeWins++;
    else if (a > h) results.awayWins++;
    else results.draws++;
    
    if (a === 0) results.homeClean++;
    if (h === 0) results.awayClean++;
    if (h > 0 && a > 0) results.btts++;
  }
  
  return {
    homeXg: lambdaH,
    awayXg: lambdaA,
    homeWin: results.homeWins / nSims,
    draw: results.draws / nSims,
    awayWin: results.awayWins / nSims,
    scores: Object.fromEntries(
      Object.entries(results.scores)
        .map(([k, v]) => [k, v / nSims])
        .sort((a, b) => b[1] - a[1])
        .slice(0, 12)
    ),
    btts: results.btts / nSims,
    over15: results.totalGoals.filter(g => g > 1.5).length / nSims,
    over25: results.totalGoals.filter(g => g > 2.5).length / nSims,
    over35: results.totalGoals.filter(g => g > 3.5).length / nSims,
    homeClean: results.homeClean / nSims,
    awayClean: results.awayClean / nSims,
  };
};

// Components
const StatBar = ({ label, value, color, showPercent = true }) => (
  <div className="mb-2">
    <div className="flex justify-between text-xs mb-1">
      <span className="text-gray-600">{label}</span>
      <span className="font-semibold" style={{ color }}>
        {showPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(2)}
      </span>
    </div>
    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
      <div 
        className="h-full rounded-full transition-all duration-700"
        style={{ 
          width: `${showPercent ? value * 100 : Math.min(value / 4, 1) * 100}%`, 
          backgroundColor: color 
        }}
      />
    </div>
  </div>
);

const TeamCard = ({ stats, side, selected, onSelect }) => {
  const bgColor = side === 'home' ? 'bg-blue-50 border-blue-200' : 'bg-red-50 border-red-200';
  const textColor = side === 'home' ? 'text-blue-800' : 'text-red-800';
  const emoji = side === 'home' ? '🏠' : '✈️';
  
  return (
    <div className={`rounded-xl p-4 border-2 ${bgColor}`}>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl">{emoji}</span>
        <span className={`font-semibold ${textColor}`}>{side === 'home' ? 'Home' : 'Away'} Team</span>
      </div>
      
      <select
        value={selected}
        onChange={(e) => onSelect(e.target.value)}
        className="w-full p-2 rounded-lg border mb-3 font-medium"
      >
        {TEAM_NAMES.map(team => (
          <option key={team} value={team}>{team}</option>
        ))}
      </select>
      
      {stats && (
        <div className="space-y-1 text-xs">
          <div className="flex justify-between py-1 border-b border-gray-200">
            <span className="text-gray-500">Elo Rating</span>
            <span className="font-bold">{stats.elo.toLocaleString()}</span>
          </div>
          <div className="flex justify-between py-1 border-b border-gray-200">
            <span className="text-gray-500">Record</span>
            <span className="font-semibold">{stats.wins}W - {stats.draws}D - {stats.losses}L</span>
          </div>
          <div className="flex justify-between py-1 border-b border-gray-200">
            <span className="text-gray-500">Goals</span>
            <span>{stats.goals_scored} scored / {stats.goals_conceded} conceded</span>
          </div>
          <div className="flex justify-between py-1 border-b border-gray-200">
            <span className="text-gray-500">xG</span>
            <span>{stats.xg_for.toFixed(1)} / {stats.xg_against.toFixed(1)}</span>
          </div>
          <div className="flex justify-between py-1 border-b border-gray-200">
            <span className="text-gray-500">Attack Strength</span>
            <span className="font-semibold text-green-600">{stats.attackStrength.toFixed(3)}</span>
          </div>
          <div className="flex justify-between py-1 border-b border-gray-200">
            <span className="text-gray-500">Defense Strength</span>
            <span className="font-semibold text-orange-600">{stats.defenseStrength.toFixed(3)}</span>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-gray-500">Form</span>
            <span className="font-mono">{stats.form.join(' ')}</span>
          </div>
        </div>
      )}
    </div>
  );
};

const ScoreHeatmap = ({ homeXg, awayXg }) => {
  const matrix = useMemo(() => getScoreMatrix(homeXg, awayXg, 5), [homeXg, awayXg]);
  const maxProb = Math.max(...matrix.flat());
  
  return (
    <div className="bg-white rounded-lg p-3">
      <h4 className="text-xs font-semibold text-gray-600 mb-2 text-center">Score Matrix</h4>
      <div className="flex justify-center">
        <div className="text-xs">
          <div className="flex">
            <div className="w-6 h-5" />
            {[0,1,2,3,4,5].map(a => (
              <div key={a} className="w-8 h-5 flex items-center justify-center text-gray-400">{a}</div>
            ))}
          </div>
          {[0,1,2,3,4,5].map(h => (
            <div key={h} className="flex">
              <div className="w-6 h-8 flex items-center justify-center text-gray-400">{h}</div>
              {[0,1,2,3,4,5].map(a => {
                const prob = matrix[h][a];
                const intensity = prob / maxProb;
                const bg = h > a 
                  ? `rgba(59, 130, 246, ${0.15 + intensity * 0.75})` 
                  : h < a 
                    ? `rgba(239, 68, 68, ${0.15 + intensity * 0.75})`
                    : `rgba(107, 114, 128, ${0.15 + intensity * 0.75})`;
                return (
                  <div 
                    key={`${h}-${a}`}
                    className="w-8 h-8 flex items-center justify-center border border-white rounded text-[10px] font-medium"
                    style={{ backgroundColor: bg, color: intensity > 0.4 ? 'white' : '#666' }}
                    title={`${h}-${a}: ${(prob*100).toFixed(2)}%`}
                  >
                    {(prob * 100).toFixed(1)}
                  </div>
                );
              })}
            </div>
          ))}
          <div className="text-center text-gray-400 mt-1">Away →</div>
        </div>
      </div>
    </div>
  );
};

// Main App
export default function StatsBombPredictor() {
  const [homeTeam, setHomeTeam] = useState('Real Madrid');
  const [awayTeam, setAwayTeam] = useState('Barcelona');
  const [simulations, setSimulations] = useState(10000);
  const [useXg, setUseXg] = useState(true);
  const [xgWeight, setXgWeight] = useState(0.6);
  const [prediction, setPrediction] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  
  const homeStats = useMemo(() => getTeamStats(homeTeam), [homeTeam]);
  const awayStats = useMemo(() => getTeamStats(awayTeam), [awayTeam]);
  
  const runPrediction = () => {
    setIsRunning(true);
    setTimeout(() => {
      const result = runSimulation(homeStats, awayStats, simulations, useXg, xgWeight);
      setPrediction({ ...result, homeTeam, awayTeam, homeElo: homeStats.elo, awayElo: awayStats.elo });
      setIsRunning(false);
    }, 50);
  };
  
  const getWinner = () => {
    if (!prediction) return null;
    if (prediction.homeWin > prediction.draw && prediction.homeWin > prediction.awayWin) 
      return { team: homeTeam, prob: prediction.homeWin, emoji: '🏠' };
    if (prediction.awayWin > prediction.draw) 
      return { team: awayTeam, prob: prediction.awayWin, emoji: '✈️' };
    return { team: 'Draw', prob: prediction.draw, emoji: '🤝' };
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-4">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold text-white mb-1 flex items-center justify-center gap-2">
            ⚽ StatsBomb Match Predictor
          </h1>
          <p className="text-blue-200 text-xs">
            Monte Carlo • Dixon-Coles • xG Model • La Liga 2020/21 Data
          </p>
          <div className="mt-2 inline-flex items-center gap-2 bg-blue-800/50 rounded-full px-3 py-1">
            <span className="text-xs text-blue-200">Powered by</span>
            <span className="text-xs font-semibold text-white">StatsBomb Open Data</span>
          </div>
        </div>
        
        {/* Team Selection */}
        <div className="bg-white/95 rounded-2xl shadow-xl p-5 mb-5">
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <TeamCard 
              stats={homeStats} 
              side="home" 
              selected={homeTeam}
              onSelect={setHomeTeam}
            />
            <TeamCard 
              stats={awayStats} 
              side="away" 
              selected={awayTeam}
              onSelect={setAwayTeam}
            />
          </div>
          
          {/* Settings */}
          <div className="grid grid-cols-3 gap-4 p-3 bg-gray-50 rounded-lg mb-4">
            <div>
              <label className="text-xs text-gray-500 block mb-1">Simulations</label>
              <select 
                value={simulations} 
                onChange={(e) => setSimulations(+e.target.value)}
                className="w-full p-2 border rounded text-sm"
              >
                <option value={1000}>1,000</option>
                <option value={10000}>10,000</option>
                <option value={50000}>50,000</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500 block mb-1">Use xG Data</label>
              <button
                onClick={() => setUseXg(!useXg)}
                className={`w-full p-2 rounded text-sm font-medium transition ${
                  useXg ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-600'
                }`}
              >
                {useXg ? '✓ Enabled' : 'Disabled'}
              </button>
            </div>
            <div>
              <label className="text-xs text-gray-500 block mb-1">xG Weight: {(xgWeight * 100).toFixed(0)}%</label>
              <input
                type="range"
                min="0"
                max="100"
                value={xgWeight * 100}
                onChange={(e) => setXgWeight(e.target.value / 100)}
                disabled={!useXg}
                className="w-full"
              />
            </div>
          </div>
          
          <button
            onClick={runPrediction}
            disabled={isRunning || homeTeam === awayTeam}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold text-lg hover:from-blue-700 hover:to-indigo-700 transition disabled:opacity-50 shadow-lg"
          >
            {isRunning ? '⏳ Running Simulation...' : '🎯 Run Prediction'}
          </button>
        </div>
        
        {/* Results */}
        {prediction && (
          <div className="bg-white/95 rounded-2xl shadow-xl overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-red-600 p-5 text-white">
              <div className="flex items-center justify-between">
                <div className="text-center flex-1">
                  <div className="text-xl font-bold">{prediction.homeTeam}</div>
                  <div className="text-blue-200 text-xs">Elo: {prediction.homeElo}</div>
                </div>
                <div className="text-center px-4">
                  <div className="text-2xl font-bold font-mono">
                    {prediction.homeXg.toFixed(2)} - {prediction.awayXg.toFixed(2)}
                  </div>
                  <div className="text-purple-200 text-xs">Expected Goals</div>
                </div>
                <div className="text-center flex-1">
                  <div className="text-xl font-bold">{prediction.awayTeam}</div>
                  <div className="text-red-200 text-xs">Elo: {prediction.awayElo}</div>
                </div>
              </div>
              
              {/* Winner Banner */}
              {getWinner() && (
                <div className="mt-4 text-center bg-white/20 rounded-lg py-2">
                  <span className="text-lg">{getWinner().emoji}</span>
                  <span className="font-bold ml-2">{getWinner().team}</span>
                  <span className="text-purple-200 ml-2">({(getWinner().prob * 100).toFixed(1)}% confidence)</span>
                </div>
              )}
            </div>
            
            <div className="p-5">
              <div className="grid md:grid-cols-2 gap-5">
                {/* Outcome Probabilities */}
                <div className="bg-gray-50 rounded-xl p-4">
                  <h3 className="font-semibold text-gray-700 mb-3 flex items-center gap-2">
                    📊 Match Outcome
                  </h3>
                  <StatBar label="🏠 Home Win" value={prediction.homeWin} color="#3B82F6" />
                  <StatBar label="🤝 Draw" value={prediction.draw} color="#6B7280" />
                  <StatBar label="✈️ Away Win" value={prediction.awayWin} color="#EF4444" />
                </div>
                
                {/* Markets */}
                <div className="bg-gray-50 rounded-xl p-4">
                  <h3 className="font-semibold text-gray-700 mb-3 flex items-center gap-2">
                    🎰 Markets
                  </h3>
                  <StatBar label="Both Teams Score" value={prediction.btts} color="#8B5CF6" />
                  <StatBar label="Over 1.5 Goals" value={prediction.over15} color="#10B981" />
                  <StatBar label="Over 2.5 Goals" value={prediction.over25} color="#F59E0B" />
                  <StatBar label="Over 3.5 Goals" value={prediction.over35} color="#EC4899" />
                </div>
              </div>
              
              <div className="grid md:grid-cols-2 gap-5 mt-5">
                {/* Score Distribution */}
                <div className="bg-gray-50 rounded-xl p-4">
                  <h3 className="font-semibold text-gray-700 mb-3">🎯 Most Likely Scores</h3>
                  <div className="space-y-1">
                    {Object.entries(prediction.scores).slice(0, 6).map(([score, prob], i) => (
                      <div key={score} className="flex items-center gap-2 text-sm">
                        <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs text-white ${
                          i === 0 ? 'bg-yellow-500' : i === 1 ? 'bg-gray-400' : i === 2 ? 'bg-amber-600' : 'bg-gray-300'
                        }`}>{i + 1}</span>
                        <span className="font-mono font-bold w-10">{score}</span>
                        <div className="flex-1 h-2 bg-gray-200 rounded-full">
                          <div className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full" 
                               style={{ width: `${prob * 100 * 4}%` }} />
                        </div>
                        <span className="text-xs text-gray-500 w-12 text-right">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Heatmap */}
                <ScoreHeatmap homeXg={prediction.homeXg} awayXg={prediction.awayXg} />
              </div>
              
              <div className="mt-4 text-center text-xs text-gray-400 pt-3 border-t">
                {simulations.toLocaleString()} Monte Carlo simulations • Dixon-Coles model with {useXg ? `xG (${(xgWeight*100).toFixed(0)}% weight)` : 'goals-only'}
              </div>
            </div>
          </div>
        )}
        
        {/* Disclaimer */}
        <div className="mt-5 text-center text-xs text-blue-200/70">
          ⚠️ For educational purposes only. Data: StatsBomb Open Data (La Liga 2020/21)
        </div>
      </div>
    </div>
  );
}
