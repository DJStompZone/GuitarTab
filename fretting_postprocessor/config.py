"""
Guitar Configuration
====================

Configuration classes and presets for guitar properties including tuning,
capo position, and fret range.
"""

from dataclasses import dataclass
from typing import Tuple, List


# Common tuning presets (MIDI pitch values)
STANDARD_TUNING = (40, 45, 50, 55, 59, 64)  # E2, A2, D3, G3, B3, E4
DROP_D_TUNING = (38, 45, 50, 55, 59, 64)    # D2, A2, D3, G3, B3, E4
HALF_STEP_DOWN = (39, 44, 49, 54, 58, 63)   # Eb2, Ab2, Db3, Gb3, Bb3, Eb4
FULL_STEP_DOWN = (38, 43, 48, 53, 57, 62)   # D2, G2, C3, F3, A3, D4


@dataclass
class GuitarConfig:
    """
    Guitar configuration including tuning, capo, and physical constraints.

    This class encapsulates all guitar-specific parameters needed for
    post-processing, including tuning, capo position, and valid fret ranges.

    Attributes:
        num_strings: Number of strings (typically 6)
        tuning: Tuple of MIDI pitches for open strings (low to high)
        capo_fret: Capo position (0 = no capo)
        min_fret: Minimum fret number (typically 0)
        max_fret: Maximum fret number (typically 24)

    Examples:
        >>> # Standard 6-string guitar
        >>> config = GuitarConfig()
        >>> config.tuning
        (40, 45, 50, 55, 59, 64)

        >>> # Drop-D with capo on 2nd fret
        >>> config = GuitarConfig(tuning=DROP_D_TUNING, capo_fret=2)
        >>> config.get_effective_tuning()
        (40, 47, 52, 57, 61, 66)

        >>> # Find all ways to play middle C (pitch 60)
        >>> positions = config.pitch_to_string_fret(60)
        >>> positions
        [(2, 10), (3, 5), (4, 0)]  # String 2 fret 10, String 3 fret 5, String 4 open
    """

    num_strings: int = 6
    tuning: Tuple[int, ...] = STANDARD_TUNING
    capo_fret: int = 0
    min_fret: int = 0
    max_fret: int = 24

    def get_effective_tuning(self) -> Tuple[int, ...]:
        """
        Get tuning adjusted for capo position.

        When a capo is placed on fret N, it effectively raises the pitch
        of each open string by N semitones.

        Returns:
            Tuple of MIDI pitches accounting for capo

        Example:
            >>> config = GuitarConfig(capo_fret=2)
            >>> config.get_effective_tuning()
            (42, 47, 52, 57, 61, 66)  # Each string raised by 2 semitones
        """
        return tuple(pitch + self.capo_fret for pitch in self.tuning)

    def is_valid_string(self, string: int) -> bool:
        """
        Check if string index is valid.

        Args:
            string: String index (0-indexed)

        Returns:
            True if string is within valid range [0, num_strings)
        """
        return 0 <= string < self.num_strings

    def is_valid_fret(self, fret: int) -> bool:
        """
        Check if fret number is valid.

        Args:
            fret: Fret number

        Returns:
            True if fret is within valid range [min_fret, max_fret]
        """
        return self.min_fret <= fret <= self.max_fret

    def pitch_to_string_fret(self, pitch: int) -> List[Tuple[int, int]]:
        """
        Find all valid (string, fret) combinations that produce the given pitch.

        This is the core function for tablature generation. For any given pitch,
        there may be multiple ways to play it on different strings.

        Args:
            pitch: Target MIDI pitch

        Returns:
            List of (string_index, fret_number) tuples, sorted by string
            (low to high). Empty list if pitch cannot be played.

        Example:
            >>> config = GuitarConfig()  # Standard tuning
            >>> config.pitch_to_string_fret(45)  # A2
            [(0, 5), (1, 0)]  # String 0 fret 5, OR string 1 open

            >>> config.pitch_to_string_fret(28)  # E1 (too low)
            []  # Cannot be played

        Note:
            Formula: fret = pitch - effective_tuning[string]
            Only returns positions where fret is in valid range.
        """
        effective_tuning = self.get_effective_tuning()
        valid_positions = []

        for string_idx, open_pitch in enumerate(effective_tuning):
            fret = pitch - open_pitch

            if self.is_valid_fret(fret):
                valid_positions.append((string_idx, fret))

        return valid_positions

    def get_pitch_range(self) -> Tuple[int, int]:
        """
        Get the playable pitch range for this guitar configuration.

        Returns:
            Tuple of (min_pitch, max_pitch)

        Example:
            >>> config = GuitarConfig()  # Standard tuning, 24 frets
            >>> config.get_pitch_range()
            (40, 88)  # E2 to E6
        """
        effective_tuning = self.get_effective_tuning()
        min_pitch = min(effective_tuning) + self.min_fret
        max_pitch = max(effective_tuning) + self.max_fret
        return (min_pitch, max_pitch)

    @classmethod
    def from_tuning_list(cls, tuning: List[int], capo: int = 0, max_fret: int = 24):
        """
        Factory method to create config from tuning list.

        Args:
            tuning: List of MIDI pitches for open strings
            capo: Capo position (default 0)
            max_fret: Maximum fret number (default 24)

        Returns:
            GuitarConfig instance

        Example:
            >>> config = GuitarConfig.from_tuning_list([40, 45, 50, 55, 59, 64])
            >>> config.num_strings
            6
        """
        return cls(
            num_strings=len(tuning),
            tuning=tuple(tuning),
            capo_fret=capo,
            max_fret=max_fret
        )

    def infer_tablature_from_pitch(
        self,
        pitch: int,
        prefer_string: int = None,
        avoid_strings: set = None
    ) -> List[Tuple[int, int]]:
        """
        從 MIDI pitch 推導所有可能的 (string, fret) 組合，按可行性排序。

        與 pitch_to_string_fret() 類似，但添加了啟發式評分以優先推薦
        更易演奏的位置。

        Args:
            pitch: MIDI pitch value
            prefer_string: 優先選擇的弦索引（如果可能）
            avoid_strings: 要避免的弦索引集合

        Returns:
            List of (string, fret) tuples，按可行性分數排序（高分優先）

        Example:
            >>> config = GuitarConfig()
            >>> # Middle C (60) 可以在多條弦上演奏
            >>> positions = config.infer_tablature_from_pitch(60)
            >>> positions[0]  # 最佳選擇（最易演奏）
            (3, 5)  # G string, 5th fret

            >>> # 偏好特定弦
            >>> positions = config.infer_tablature_from_pitch(60, prefer_string=2)
            >>> positions[0]
            (2, 10)  # D string, 10th fret (因為被偏好)

        Scoring heuristics:
            - Strong preference for specified string (+100 points)
            - Preference for middle frets (7th fret = optimal)
            - Preference for middle strings (avoid extreme highs/lows)
        """
        effective_tuning = self.get_effective_tuning()
        possibilities = []

        for string in range(len(effective_tuning)):
            # 跳過要避免的弦
            if avoid_strings and string in avoid_strings:
                continue

            fret = pitch - effective_tuning[string]

            if self.is_valid_fret(fret):
                # 計算可行性分數
                score = 0.0

                # 強烈偏好指定的弦
                if prefer_string is not None and string == prefer_string:
                    score += 100

                # 偏好中間品格（第7品附近最佳）
                fret_distance = abs(fret - 7)
                score -= fret_distance * 0.5

                # 偏好中間弦（避免極端高/低弦）
                string_center = (self.num_strings - 1) / 2.0
                string_distance = abs(string - string_center)
                score -= string_distance * 1.0

                # 避免開放弦（0品），稍微降低分數
                if fret == 0:
                    score -= 2

                # 避免過高品格（>12），顯著降低分數
                if fret > 12:
                    score -= (fret - 12) * 2

                possibilities.append((string, fret, score))

        # 按分數排序（高分優先）
        possibilities.sort(key=lambda x: x[2], reverse=True)

        # 返回 (string, fret) 對，移除分數
        return [(s, f) for s, f, _ in possibilities]

    def get_default_tablature_for_pitch(self, pitch: int) -> Tuple[int, int]:
        """
        獲取 pitch 的預設 (string, fret) 位置。

        這是 infer_tablature_from_pitch() 的簡化版本，直接返回最佳選擇。
        適用於快速推導單個 note 的 tablature。

        Args:
            pitch: MIDI pitch value

        Returns:
            (string, fret) tuple 代表最佳演奏位置
            如果 pitch 無法在吉他上演奏則返回 None

        Example:
            >>> config = GuitarConfig()
            >>> config.get_default_tablature_for_pitch(60)  # Middle C
            (3, 5)  # G string, 5th fret

            >>> config.get_default_tablature_for_pitch(20)  # Too low
            None  # 無法演奏
        """
        possibilities = self.infer_tablature_from_pitch(pitch)

        if possibilities:
            return possibilities[0]  # 返回分數最高的選擇
        else:
            return None  # 無法演奏此 pitch

    def __repr__(self) -> str:
        """String representation for debugging"""
        tuning_str = ', '.join(str(p) for p in self.tuning)
        capo_str = f", capo={self.capo_fret}" if self.capo_fret > 0 else ""
        return f"GuitarConfig({self.num_strings} strings, tuning=({tuning_str}){capo_str})"
