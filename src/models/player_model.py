"""
Player-Level Simulation Model

Models the impact of individual players on team performance including:
- Squad strength calculation
- Injury/suspension impact
- Formation effectiveness
- Key player dependency
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum


class Position(Enum):
    """Player positions."""
    GK = "Goalkeeper"
    CB = "Center Back"
    LB = "Left Back"
    RB = "Right Back"
    CDM = "Defensive Midfielder"
    CM = "Central Midfielder"
    CAM = "Attacking Midfielder"
    LW = "Left Winger"
    RW = "Right Winger"
    CF = "Center Forward"
    ST = "Striker"


@dataclass
class Player:
    """Individual player representation."""
    name: str
    position: Position
    overall_rating: float  # 0-100 scale
    attack_rating: float = 0.0
    defense_rating: float = 0.0
    physical_rating: float = 0.0
    mental_rating: float = 0.0
    is_key_player: bool = False
    injury_prone: float = 0.1  # Probability weight for injuries
    
    def __post_init__(self):
        # Auto-calculate sub-ratings if not provided
        if self.attack_rating == 0:
            if self.position in [Position.ST, Position.CF, Position.LW, Position.RW, Position.CAM]:
                self.attack_rating = self.overall_rating * 1.1
            else:
                self.attack_rating = self.overall_rating * 0.8
                
        if self.defense_rating == 0:
            if self.position in [Position.GK, Position.CB, Position.LB, Position.RB, Position.CDM]:
                self.defense_rating = self.overall_rating * 1.1
            else:
                self.defense_rating = self.overall_rating * 0.7


@dataclass
class Formation:
    """Team formation with position weights."""
    name: str
    positions: Dict[Position, int]  # Position -> count
    attack_weight: float = 1.0
    defense_weight: float = 1.0
    midfield_control: float = 1.0


# Common formations
FORMATIONS = {
    "4-3-3": Formation(
        name="4-3-3",
        positions={
            Position.GK: 1, Position.CB: 2, Position.LB: 1, Position.RB: 1,
            Position.CM: 3, Position.LW: 1, Position.RW: 1, Position.ST: 1
        },
        attack_weight=1.1,
        defense_weight=0.95,
        midfield_control=1.0
    ),
    "4-4-2": Formation(
        name="4-4-2",
        positions={
            Position.GK: 1, Position.CB: 2, Position.LB: 1, Position.RB: 1,
            Position.CM: 2, Position.LW: 1, Position.RW: 1, Position.ST: 2
        },
        attack_weight=1.05,
        defense_weight=1.0,
        midfield_control=0.95
    ),
    "4-2-3-1": Formation(
        name="4-2-3-1",
        positions={
            Position.GK: 1, Position.CB: 2, Position.LB: 1, Position.RB: 1,
            Position.CDM: 2, Position.CAM: 1, Position.LW: 1, Position.RW: 1, Position.ST: 1
        },
        attack_weight=1.0,
        defense_weight=1.05,
        midfield_control=1.1
    ),
    "3-5-2": Formation(
        name="3-5-2",
        positions={
            Position.GK: 1, Position.CB: 3,
            Position.CDM: 2, Position.CM: 1, Position.LW: 1, Position.RW: 1, Position.ST: 2
        },
        attack_weight=1.15,
        defense_weight=0.9,
        midfield_control=1.05
    ),
    "5-3-2": Formation(
        name="5-3-2",
        positions={
            Position.GK: 1, Position.CB: 3, Position.LB: 1, Position.RB: 1,
            Position.CM: 3, Position.ST: 2
        },
        attack_weight=0.9,
        defense_weight=1.2,
        midfield_control=0.95
    ),
    "4-1-4-1": Formation(
        name="4-1-4-1",
        positions={
            Position.GK: 1, Position.CB: 2, Position.LB: 1, Position.RB: 1,
            Position.CDM: 1, Position.CM: 2, Position.LW: 1, Position.RW: 1, Position.ST: 1
        },
        attack_weight=0.95,
        defense_weight=1.1,
        midfield_control=1.05
    )
}


@dataclass
class Squad:
    """Team squad with players and formation."""
    name: str
    players: List[Player] = field(default_factory=list)
    formation: str = "4-3-3"
    injured: Set[str] = field(default_factory=set)
    suspended: Set[str] = field(default_factory=set)
    
    @property
    def available_players(self) -> List[Player]:
        """Get list of available (non-injured, non-suspended) players."""
        unavailable = self.injured | self.suspended
        return [p for p in self.players if p.name not in unavailable]
    
    @property
    def starting_xi(self) -> List[Player]:
        """Get best starting XI based on formation."""
        formation = FORMATIONS.get(self.formation, FORMATIONS["4-3-3"])
        available = self.available_players
        
        xi = []
        for position, count in formation.positions.items():
            # Get best players for this position
            position_players = sorted(
                [p for p in available if p.position == position],
                key=lambda x: x.overall_rating,
                reverse=True
            )[:count]
            
            # Fill with similar positions if needed
            while len(position_players) < count and available:
                remaining = [p for p in available if p not in xi and p not in position_players]
                if remaining:
                    position_players.append(max(remaining, key=lambda x: x.overall_rating))
                else:
                    break
                    
            xi.extend(position_players)
            
        return xi[:11]


class PlayerModel:
    """
    Model for calculating team strength based on player-level data.
    
    Features:
    - Squad strength aggregation
    - Injury/suspension impact modeling
    - Formation effectiveness
    - Key player dependency analysis
    """
    
    def __init__(
        self,
        key_player_impact: float = 0.15,
        position_importance: Optional[Dict[Position, float]] = None,
        injury_impact_scale: float = 0.1
    ):
        """
        Initialize player model.
        
        Args:
            key_player_impact: Extra weight for key players' absence
            position_importance: Custom position importance weights
            injury_impact_scale: How much each missing player affects xG
        """
        self.key_player_impact = key_player_impact
        self.injury_impact_scale = injury_impact_scale
        
        self.position_importance = position_importance or {
            Position.GK: 1.2,
            Position.CB: 1.0,
            Position.LB: 0.8,
            Position.RB: 0.8,
            Position.CDM: 1.0,
            Position.CM: 1.0,
            Position.CAM: 1.1,
            Position.LW: 0.9,
            Position.RW: 0.9,
            Position.CF: 1.1,
            Position.ST: 1.2
        }
        
    def calculate_squad_strength(
        self,
        squad: Squad,
        include_bench: bool = False
    ) -> Dict[str, float]:
        """
        Calculate overall squad strength.
        
        Args:
            squad: Squad object with players
            include_bench: Whether to include bench depth
            
        Returns:
            Dictionary with strength metrics
        """
        xi = squad.starting_xi
        
        if not xi:
            return {
                'overall': 50.0,
                'attack': 50.0,
                'defense': 50.0,
                'depth': 0.0
            }
        
        # Weighted average by position importance
        total_weight = 0
        weighted_overall = 0
        weighted_attack = 0
        weighted_defense = 0
        
        for player in xi:
            weight = self.position_importance.get(player.position, 1.0)
            total_weight += weight
            weighted_overall += player.overall_rating * weight
            weighted_attack += player.attack_rating * weight
            weighted_defense += player.defense_rating * weight
        
        overall = weighted_overall / total_weight if total_weight > 0 else 50
        attack = weighted_attack / total_weight if total_weight > 0 else 50
        defense = weighted_defense / total_weight if total_weight > 0 else 50
        
        # Apply formation modifiers
        formation = FORMATIONS.get(squad.formation, FORMATIONS["4-3-3"])
        attack *= formation.attack_weight
        defense *= formation.defense_weight
        
        # Squad depth (bench quality)
        bench = [p for p in squad.available_players if p not in xi]
        depth = np.mean([p.overall_rating for p in bench]) if bench else 0
        
        return {
            'overall': overall,
            'attack': attack,
            'defense': defense,
            'depth': depth,
            'formation_attack_mod': formation.attack_weight,
            'formation_defense_mod': formation.defense_weight
        }
    
    def injury_impact(
        self,
        squad: Squad,
        injured_names: Optional[Set[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate the impact of injuries on team strength.
        
        Args:
            squad: Squad object
            injured_names: Set of injured player names (uses squad.injured if None)
            
        Returns:
            Dictionary with impact factors
        """
        if injured_names is None:
            injured_names = squad.injured
            
        injured_players = [p for p in squad.players if p.name in injured_names]
        
        if not injured_players:
            return {
                'attack_factor': 1.0,
                'defense_factor': 1.0,
                'overall_factor': 1.0,
                'key_player_missing': False
            }
        
        # Calculate impact based on player importance
        attack_loss = 0
        defense_loss = 0
        overall_loss = 0
        key_player_missing = False
        
        for player in injured_players:
            importance = self.position_importance.get(player.position, 1.0)
            
            attack_loss += player.attack_rating * importance * self.injury_impact_scale
            defense_loss += player.defense_rating * importance * self.injury_impact_scale
            overall_loss += player.overall_rating * importance * self.injury_impact_scale
            
            if player.is_key_player:
                key_player_missing = True
                attack_loss += self.key_player_impact * 100
                defense_loss += self.key_player_impact * 100
        
        # Convert to multiplicative factors (cap at 50% reduction)
        attack_factor = max(0.5, 1.0 - attack_loss / 100)
        defense_factor = max(0.5, 1.0 - defense_loss / 100)
        overall_factor = max(0.5, 1.0 - overall_loss / 100)
        
        return {
            'attack_factor': attack_factor,
            'defense_factor': defense_factor,
            'overall_factor': overall_factor,
            'key_player_missing': key_player_missing
        }
    
    def formation_matchup(
        self,
        home_formation: str,
        away_formation: str
    ) -> Dict[str, float]:
        """
        Calculate formation matchup advantages.
        
        Args:
            home_formation: Home team formation string
            away_formation: Away team formation string
            
        Returns:
            Dictionary with matchup factors
        """
        home_f = FORMATIONS.get(home_formation, FORMATIONS["4-3-3"])
        away_f = FORMATIONS.get(away_formation, FORMATIONS["4-3-3"])
        
        # Midfield control advantage
        midfield_diff = home_f.midfield_control - away_f.midfield_control
        
        # Attack vs defense mismatches
        home_attack_vs_defense = home_f.attack_weight / away_f.defense_weight
        away_attack_vs_defense = away_f.attack_weight / home_f.defense_weight
        
        return {
            'home_midfield_advantage': midfield_diff,
            'home_attack_efficiency': home_attack_vs_defense,
            'away_attack_efficiency': away_attack_vs_defense
        }
    
    def expected_goals_adjustment(
        self,
        base_xg: float,
        squad: Squad,
        is_attacking: bool = True
    ) -> float:
        """
        Adjust expected goals based on player factors.
        
        Args:
            base_xg: Base expected goals from other models
            squad: Squad object
            is_attacking: True for attack xG, False for defense
            
        Returns:
            Adjusted expected goals
        """
        strength = self.calculate_squad_strength(squad)
        injury_impact = self.injury_impact(squad)
        
        if is_attacking:
            # Normalize attack strength (75 = average)
            strength_factor = strength['attack'] / 75.0
            injury_factor = injury_impact['attack_factor']
        else:
            # For defense, higher = better, so invert for goals against
            strength_factor = 75.0 / strength['defense']
            injury_factor = injury_impact['defense_factor']
        
        # Apply adjustments
        adjusted_xg = base_xg * strength_factor * injury_factor
        
        return max(0.2, adjusted_xg)  # Minimum 0.2 xG
    
    def simulate_player_events(
        self,
        squad: Squad,
        n_goals: int
    ) -> List[str]:
        """
        Simulate which players score in a match.
        
        Args:
            squad: Squad object
            n_goals: Number of goals to attribute
            
        Returns:
            List of scorer names
        """
        xi = squad.starting_xi
        if not xi or n_goals == 0:
            return []
        
        # Weight by attack rating and position
        weights = []
        for player in xi:
            weight = player.attack_rating / 100
            if player.position in [Position.ST, Position.CF]:
                weight *= 2.5
            elif player.position in [Position.LW, Position.RW, Position.CAM]:
                weight *= 1.5
            elif player.position == Position.CM:
                weight *= 0.8
            else:
                weight *= 0.3
            weights.append(weight)
        
        # Normalize
        total = sum(weights)
        probs = [w / total for w in weights]
        
        # Sample scorers
        scorer_indices = np.random.choice(
            len(xi), size=n_goals, p=probs, replace=True
        )
        
        return [xi[i].name for i in scorer_indices]
    
    def create_squad_from_dict(
        self,
        data: Dict
    ) -> Squad:
        """
        Create Squad object from dictionary data.
        
        Args:
            data: Dictionary with squad data
            
        Returns:
            Squad object
        """
        players = []
        for p_data in data.get('players', []):
            position_str = p_data.get('position', 'CM')
            position = Position[position_str] if position_str in Position.__members__ else Position.CM
            
            players.append(Player(
                name=p_data['name'],
                position=position,
                overall_rating=p_data.get('rating', 70),
                attack_rating=p_data.get('attack', 0),
                defense_rating=p_data.get('defense', 0),
                is_key_player=p_data.get('key_player', False)
            ))
        
        return Squad(
            name=data.get('name', 'Unknown'),
            players=players,
            formation=data.get('formation', '4-3-3'),
            injured=set(data.get('injuries', [])),
            suspended=set(data.get('suspended', []))
        )
