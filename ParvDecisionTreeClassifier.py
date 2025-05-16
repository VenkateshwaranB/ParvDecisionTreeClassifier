import numpy as np
import pandas as pd
import re
from collections import defaultdict, OrderedDict, Counter, defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from functools import lru_cache

class ParvNode:
    """A node in the Parv Decision Tree"""
    def __init__(self, depth=0):
        self.depth = depth
        self.feature_idx = None
        self.feature_value = None
        self.threshold = None  # For continuous features
        self.feature_type = None  # 'categorical' or 'continuous'
        self.class_level = None
        self.children = {}
        self.leaf = False
        self.prediction = None
        self.Parv_gain = None
        self.traditional_gain = None
        self.hybrid_gain = None
        self.samples = 0
        self.class_distribution = None

class ClassExtractor:
    """Class for extracting hierarchical class levels from feature values"""
    
    def __init__(self, method='prefix', prefix_length=1, pattern=None, levels=None):
        """
        Initialize the class extractor
        
        Parameters:
        -----------
        method : str, default='prefix'
            Method to extract class levels: 'prefix', 'suffix', 'regex', 'custom'
        
        prefix_length : int, default=1
            Number of characters to remove when using 'prefix' method
            
        pattern : str, default=None
            Regex pattern to use when using 'regex' method
            
        levels : list, default=None
            Ordered list of class levels from highest to lowest importance
        """
        self.method = method
        self.prefix_length = prefix_length
        self.pattern = pattern
        self.levels = levels
        self._compiled_pattern = re.compile(pattern) if pattern else None
    
    def extract(self, value):
        """Extract class level from a feature value"""
        if not isinstance(value, str):
            return value
            
        if self.method == 'prefix':
            if len(value) > self.prefix_length:
                return value[self.prefix_length:]
            return value
            
        elif self.method == 'suffix':
            if len(value) > self.prefix_length:
                return value[:-self.prefix_length]
            return value
            
        elif self.method == 'regex' and self._compiled_pattern:
            match = self._compiled_pattern.search(value)
            if match:
                return match.group(1)  # Return the first captured group
            return value
            
        # Default case
        return value
    
    def get_level_importance(self, level):
        """Get the importance of a class level based on its position in hierarchy"""
        if not self.levels or level not in self.levels:
            return 0
        
        # Reverse the index so higher levels have higher importance
        return len(self.levels) - self.levels.index(level)
    
    @staticmethod
    def detect_optimal_method(X):
        """Automatically detect the best method and parameters for class extraction"""
        # Sample feature values
        sampled_values = []
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        
        for feature_idx in range(min(n_features, 10)):
            feature_values = X[:, feature_idx] if hasattr(X, 'shape') else [row[feature_idx] for row in X]
            unique_vals = set(val for val in feature_values if isinstance(val, str))
            sampled_values.extend(list(unique_vals)[:5])
        
        # Check prefix patterns
        prefix_counts = defaultdict(int)
        for value in sampled_values:
            for i in range(1, min(len(value), 3)):
                # Check for alphabetical pattern changes
                if value[i].isalpha() and value[i-1].isalpha() and value[i-1].isupper() != value[i].isupper():
                    prefix_counts[i] += 1
        
        # Check common regex patterns
        regex_patterns = [
            (r'([A-Z]+)[^A-Z].*', 'Initial caps'),
            (r'.*[^A-Z]([A-Z]+)', 'Trailing caps'),
            (r'([A-Za-z]+)\d+.*', 'Letters followed by numbers')
        ]
        
        regex_counts = defaultdict(int)
        for value in sampled_values:
            for pattern, desc in regex_patterns:
                if re.search(pattern, value):
                    regex_counts[pattern] += 1
        
        # Determine best method
        if prefix_counts:
            best_prefix_length = max(prefix_counts.items(), key=lambda x: x[1])[0]
            return 'prefix', best_prefix_length, None
        
        if regex_counts:
            best_pattern = max(regex_counts.items(), key=lambda x: x[1])[0]
            return 'regex', None, best_pattern
        
        # Default
        return 'prefix', 1, None


class ParvDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Decision Tree Classifier that uses Paleo Affinity Index (Parv) for splitting.
    
    This classifier works with both categorical and continuous features, and
    leverages hierarchical class information to make better splitting decisions.
    
    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of the tree.
    
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
        
    class_extraction : str or dict, default='auto'
        Method to extract class levels:
        - 'auto': Automatically detect the best method
        - 'prefix': Remove first N characters to get class level
        - 'suffix': Remove last N characters to get class level
        - 'regex': Use regex pattern to extract class level
        - dict: Detailed configuration with 'method', 'prefix_length', 'pattern'
    
    class_hierarchy : list, default=None
        Ordered list of class levels from highest to lowest importance.
        
    alpha : float, default=0.5
        Mixing parameter for hybrid gain calculation:
        - 0.0: Use only traditional information gain
        - 1.0: Use only Parv gain
        - Between 0 and 1: Use weighted combination
    
    class_weight : dict or 'balanced', default=None
        Weights associated with classes:
        - None: All classes have weight one
        - 'balanced': Class weights are automatically adjusted to be inversely
          proportional to class frequencies
        - dict: Dictionary mapping class labels to class weights
    
    random_state : int, default=None
        Controls the randomness of the estimator.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 class_extraction='auto', class_hierarchy=None, alpha=0.5,
                 class_weight=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_extraction = class_extraction
        self.class_hierarchy = class_hierarchy
        self.alpha = alpha
        self.class_weight = class_weight
        self.random_state = random_state
        
        # Internal attributes
        self._computation_cache = {}  # Cache for expensive computations
        
    def _setup_class_extractor(self, X):
        """Set up the class extractor based on configuration"""
        if self.class_extraction == 'auto':
            # Automatically detect the best method
            method, prefix_length, pattern = ClassExtractor.detect_optimal_method(X)
            return ClassExtractor(method=method, prefix_length=prefix_length, 
                                  pattern=pattern, levels=self.class_hierarchy)
        
        elif isinstance(self.class_extraction, dict):
            # Use detailed configuration
            return ClassExtractor(
                method=self.class_extraction.get('method', 'prefix'),
                prefix_length=self.class_extraction.get('prefix_length', 1),
                pattern=self.class_extraction.get('pattern', None),
                levels=self.class_hierarchy
            )
        
        elif self.class_extraction == 'prefix':
            # Use prefix method with default length
            return ClassExtractor(method='prefix', prefix_length=1, levels=self.class_hierarchy)
        
        elif self.class_extraction == 'suffix':
            # Use suffix method with default length
            return ClassExtractor(method='suffix', prefix_length=1, levels=self.class_hierarchy)
        
        elif self.class_extraction == 'regex':
            # Use regex method with default pattern
            return ClassExtractor(method='regex', pattern=r'([A-Za-z]+)', levels=self.class_hierarchy)
        
        else:
            # Default to prefix method
            return ClassExtractor(method='prefix', prefix_length=1, levels=self.class_hierarchy)
    
    def _setup_class_weights(self, y):
        """Set up class weights"""
        if self.class_weight == 'balanced':
            # Compute balanced class weights
            classes = np.unique(y)
            class_counts = Counter(y)
            n_samples = len(y)
            
            # Balanced weights are inversely proportional to class frequencies
            return {cls: n_samples / (len(classes) * class_counts[cls]) for cls in classes}
        
        elif isinstance(self.class_weight, dict):
            # Use provided class weights
            return self.class_weight
        
        # Default: equal weights
        return None
        
    def _is_categorical(self, values):
        """Determine if a feature is categorical or continuous"""
        # If values are strings, feature is categorical
        if all(isinstance(v, str) for v in values if v is not None):
            return True
        
        # If values are numeric and few unique values, feature is likely categorical
        unique_values = set(values)
        if len(unique_values) <= 10 and all(isinstance(v, (int, float)) for v in unique_values if v is not None):
            return True
        
        # Otherwise, treat as continuous
        return False
    
    def _calculate_weighted_gini(self, y, sample_weight=None):
        """Calculate Gini impurity with optional sample weights"""
        if len(y) == 0:
            return 0.0
        
        # Count classes with weighting
        if sample_weight is None:
            sample_weight = np.ones(len(y))
            
        classes, counts = np.unique(y, return_counts=True)
        weighted_counts = {}
        total_weight = sum(sample_weight)
        
        for cls, count in zip(classes, counts):
            # Sum the weights for this class
            mask = y == cls
            weighted_counts[cls] = sum(sample_weight[mask])
        
        # Calculate Gini
        gini = 1.0
        for weight in weighted_counts.values():
            if total_weight > 0:
                p = weight / total_weight
                gini -= p * p
        
        return gini
    
    def _calculate_feature_gini(self, X, y, feature_idx, sample_weight=None):
        """Calculate Gini impurity for each value of a feature"""
        is_categorical = self._is_categorical(X[:, feature_idx])
        
        if is_categorical:
            # Process categorical feature
            feature_values = np.unique(X[:, feature_idx])
            value_ginis = {}
            weighted_gini = 0.0
            
            for value in feature_values:
                mask = X[:, feature_idx] == value
                subset_y = y[mask]
                
                if len(subset_y) == 0:
                    continue
                
                # Calculate Gini for this value
                subset_weights = None if sample_weight is None else sample_weight[mask]
                gini = self._calculate_weighted_gini(subset_y, subset_weights)
                
                # Calculate weight for this value
                if sample_weight is None:
                    weight = len(subset_y) / len(y)
                else:
                    weight = sum(sample_weight[mask]) / sum(sample_weight)
                
                weighted_gini += weight * gini
                
                # Extract class level
                class_level = self.class_extractor.extract(value)
                
                value_ginis[value] = {
                    'gini': gini,
                    'count': len(subset_y),
                    'weight': weight,
                    'class_level': class_level
                }
            
            return {
                'is_categorical': True,
                'value_ginis': value_ginis,
                'weighted_gini': weighted_gini
            }
            
        else:
            # Process continuous feature
            # Sort values and find optimal splits
            sorted_indices = np.argsort(X[:, feature_idx])
            sorted_values = X[sorted_indices, feature_idx]
            sorted_y = y[sorted_indices]
            sorted_weights = None if sample_weight is None else sample_weight[sorted_indices]
            
            # Find potential thresholds (midpoints between consecutive values)
            thresholds = []
            for i in range(len(sorted_values)-1):
                if sorted_values[i] != sorted_values[i+1]:
                    thresholds.append((sorted_values[i] + sorted_values[i+1])/2)
            
            if not thresholds:
                # No valid thresholds found
                return {
                    'is_categorical': False,
                    'thresholds': [],
                    'threshold_ginis': {},
                    'best_threshold': None,
                    'weighted_gini': self._calculate_weighted_gini(y, sample_weight)
                }
            
            # Calculate Gini for each threshold
            threshold_ginis = {}
            best_threshold = None
            best_weighted_gini = float('inf')
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                # Calculate Gini for left and right splits
                left_weights = None if sample_weight is None else sample_weight[left_mask]
                right_weights = None if sample_weight is None else sample_weight[right_mask]
                
                left_gini = self._calculate_weighted_gini(left_y, left_weights)
                right_gini = self._calculate_weighted_gini(right_y, right_weights)
                
                # Calculate weights for splits
                if sample_weight is None:
                    left_weight = len(left_y) / len(y)
                    right_weight = len(right_y) / len(y)
                else:
                    left_weight = sum(sample_weight[left_mask]) / sum(sample_weight)
                    right_weight = sum(sample_weight[right_mask]) / sum(sample_weight)
                
                # Calculate weighted Gini
                threshold_weighted_gini = left_weight * left_gini + right_weight * right_gini
                
                threshold_ginis[threshold] = {
                    'left_gini': left_gini,
                    'right_gini': right_gini,
                    'left_weight': left_weight,
                    'right_weight': right_weight,
                    'weighted_gini': threshold_weighted_gini
                }
                
                # Update best threshold
                if threshold_weighted_gini < best_weighted_gini:
                    best_weighted_gini = threshold_weighted_gini
                    best_threshold = threshold
            
            return {
                'is_categorical': False,
                'thresholds': thresholds,
                'threshold_ginis': threshold_ginis,
                'best_threshold': best_threshold,
                'weighted_gini': best_weighted_gini if best_threshold is not None else self._calculate_weighted_gini(y, sample_weight)
            }
    
    def _calculate_Parv(self, X, y, feature_gini_results, sample_weight=None):
        """Calculate Parv for class levels across features"""
        n_features = X.shape[1]
        parent_gini = self._calculate_weighted_gini(y, sample_weight)
        
        # Collect class level data across features
        class_level_data = defaultdict(lambda: {'gini_sum': 0.0, 'count': 0, 'features': []})
        
        # Process each feature
        for feature_idx in range(n_features):
            result = feature_gini_results[feature_idx]
            
            if result['is_categorical']:
                # Process categorical feature
                for value, data in result['value_ginis'].items():
                    class_level = data['class_level']
                    
                    # Accumulate Gini for this class level
                    class_level_data[class_level]['gini_sum'] += data['gini'] * data['weight']
                    
                    # Track features with this class level
                    if feature_idx not in class_level_data[class_level]['features']:
                        class_level_data[class_level]['features'].append(feature_idx)
                        class_level_data[class_level]['count'] += 1
            else:
                # Process continuous feature
                # For continuous features, we don't have class levels directly
                # We could assign class levels based on threshold ranges
                pass
        
        # Calculate Parv for each class level
        class_level_Parv = {}
        for class_level, data in class_level_data.items():
            if data['count'] > 0:
                # Parv(Class=Attribute) = Gini(Class) * Weighted_Gini(Attribute) / Attribute_count
                class_level_Parv[class_level] = parent_gini * data['gini_sum'] / data['count']
            else:
                class_level_Parv[class_level] = 0.0
        
        # Calculate total Parv
        total_Parv = sum(class_level_Parv.values())
        
        return class_level_Parv, total_Parv
    
    def _calculate_hybrid_gain(self, X, y, feature_idx, parent_gini, parent_Parv, sample_weight=None):
        """Calculate hybrid gain for a feature using both Parv and traditional metrics"""
        # Calculate traditional gain based on Gini impurity
        feature_result = self._calculate_feature_gini(X, y, feature_idx, sample_weight)
        
        if feature_result['is_categorical']:
            # Categorical feature
            weighted_gini = feature_result['weighted_gini']
            traditional_gain = parent_gini - weighted_gini
            
            # Calculate Parv gain (simplified approximation)
            # For more accurate Parv gain, we would need to recalculate Parv for the split subsets
            # This is a simplified version that estimates Parv gain by comparing the class levels
            Parv_gain = 0.0
            for value, data in feature_result['value_ginis'].items():
                class_level = data['class_level']
                if class_level in self.class_importance:
                    # Weight by class level importance
                    importance = self.class_importance[class_level]
                    Parv_gain += data['weight'] * importance * traditional_gain
            
            # Combine gains using alpha parameter
            hybrid_gain = self.alpha * Parv_gain + (1 - self.alpha) * traditional_gain
            
            return {
                'feature_idx': feature_idx,
                'is_categorical': True, 
                'traditional_gain': traditional_gain,
                'Parv_gain': Parv_gain,
                'hybrid_gain': hybrid_gain,
                'feature_result': feature_result
            }
            
        else:
            # Continuous feature
            if feature_result['best_threshold'] is None:
                return {
                    'feature_idx': feature_idx,
                    'is_categorical': False,
                    'traditional_gain': 0.0,
                    'Parv_gain': 0.0,
                    'hybrid_gain': 0.0,
                    'feature_result': feature_result
                }
            
            weighted_gini = feature_result['weighted_gini']
            traditional_gain = parent_gini - weighted_gini
            
            # For continuous features, we use traditional gain directly
            # since class levels aren't directly applicable
            Parv_gain = traditional_gain  # Simplified
            
            # Combine gains using alpha parameter
            hybrid_gain = self.alpha * Parv_gain + (1 - self.alpha) * traditional_gain
            
            return {
                'feature_idx': feature_idx,
                'is_categorical': False,
                'traditional_gain': traditional_gain,
                'Parv_gain': Parv_gain,
                'hybrid_gain': hybrid_gain,
                'feature_result': feature_result
            }
    
    def _find_best_split(self, X, y, sample_weight=None):
        """Find the best feature and value to split on using hybrid gain"""
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None
            
        parent_gini = self._calculate_weighted_gini(y, sample_weight)
        
        if parent_gini == 0.0:  # Pure node
            return None
        
        # Calculate Gini for each feature
        feature_gini_results = {}
        for feature_idx in range(n_features):
            feature_gini_results[feature_idx] = self._calculate_feature_gini(X, y, feature_idx, sample_weight)
        
        # Calculate Parv for class levels
        class_level_Parv, parent_Parv = self._calculate_Parv(X, y, feature_gini_results, sample_weight)
        
        # Calculate hybrid gain for each feature
        feature_gains = []
        for feature_idx in range(n_features):
            gain_result = self._calculate_hybrid_gain(X, y, feature_idx, parent_gini, parent_Parv, sample_weight)
            feature_gains.append(gain_result)
        
        # Find best feature
        best_feature = None
        best_gain = -float('inf')
        
        for gain_result in feature_gains:
            if gain_result['hybrid_gain'] > best_gain:
                best_gain = gain_result['hybrid_gain']
                best_feature = gain_result
        
        if best_feature is None or best_gain <= 0:
            return None
        
        # Prepare the split information
        split_info = {
            'feature_idx': best_feature['feature_idx'],
            'is_categorical': best_feature['is_categorical'],
            'traditional_gain': best_feature['traditional_gain'],
            'Parv_gain': best_feature['Parv_gain'],
            'hybrid_gain': best_feature['hybrid_gain']
        }
        
        if best_feature['is_categorical']:
            # For categorical features, find the best value
            feature_result = best_feature['feature_result']
            value_ginis = feature_result['value_ginis']
            
            # For categorical features, we can either:
            # 1. Split into multiple children (one per value)
            # 2. Binary split based on one value vs. all others
            
            # Option 1: Multiple children (used here)
            split_info['values'] = list(value_ginis.keys())
            split_info['class_levels'] = {value: data['class_level'] for value, data in value_ginis.items()}
            
        else:
            # For continuous features, use the best threshold
            feature_result = best_feature['feature_result']
            split_info['threshold'] = feature_result['best_threshold']
        
        return split_info
    
    def _split_node(self, X, y, split_info, sample_weight=None):
        """Split the data according to the split information"""
        feature_idx = split_info['feature_idx']
        
        if split_info['is_categorical']:
            # Categorical split - one child per value
            children = {}
            for value in split_info['values']:
                mask = X[:, feature_idx] == value
                X_child = X[mask]
                y_child = y[mask]
                
                if sample_weight is not None:
                    w_child = sample_weight[mask]
                else:
                    w_child = None
                
                if len(y_child) >= self.min_samples_leaf:
                    children[value] = {
                        'X': X_child,
                        'y': y_child,
                        'sample_weight': w_child,
                        'class_level': split_info['class_levels'].get(value)
                    }
            
            return children
            
        else:
            # Continuous split - binary split on threshold
            threshold = split_info['threshold']
            
            # Left child: <= threshold
            left_mask = X[:, feature_idx] <= threshold
            X_left = X[left_mask]
            y_left = y[left_mask]
            w_left = None if sample_weight is None else sample_weight[left_mask]
            
            # Right child: > threshold
            right_mask = ~left_mask
            X_right = X[right_mask]
            y_right = y[right_mask]
            w_right = None if sample_weight is None else sample_weight[right_mask]
            
            children = {}
            
            if len(y_left) >= self.min_samples_leaf:
                children['left'] = {
                    'X': X_left,
                    'y': y_left,
                    'sample_weight': w_left,
                    'condition': f'<= {threshold}'
                }
            
            if len(y_right) >= self.min_samples_leaf:
                children['right'] = {
                    'X': X_right,
                    'y': y_right,
                    'sample_weight': w_right,
                    'condition': f'> {threshold}'
                }
            
            return children
    
    def _build_tree(self, X, y, depth=0, sample_weight=None):
        """Recursively build the Parv decision tree"""
        node = ParvNode(depth=depth)
        node.samples = len(y)
        
        # Calculate class distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        node.class_distribution = dict(zip(unique_classes, counts))
        
        # Get majority class for leaf node prediction
        if sample_weight is None:
            # Unweighted majority
            majority_class = Counter(y).most_common(1)[0][0]
        else:
            # Weighted majority
            class_weights = {}
            for cls in unique_classes:
                mask = y == cls
                class_weights[cls] = sum(sample_weight[mask])
            majority_class = max(class_weights, key=class_weights.get)
            
        node.prediction = majority_class
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split:
            node.leaf = True
            return node
            
        # Find best split
        split_info = self._find_best_split(X, y, sample_weight)
        
        if split_info is None:  # No good split found
            node.leaf = True
            return node
            
        # Set node attributes
        node.feature_idx = split_info['feature_idx']
        node.feature_type = 'categorical' if split_info['is_categorical'] else 'continuous'
        node.traditional_gain = split_info['traditional_gain']
        node.Parv_gain = split_info['Parv_gain']
        node.hybrid_gain = split_info['hybrid_gain']
        
        if not split_info['is_categorical']:
            node.threshold = split_info['threshold']
        
        # Split the data
        children_data = self._split_node(X, y, split_info, sample_weight)
        
        # Check if split resulted in valid children
        if not children_data:
            node.leaf = True
            return node
            
        # Create child nodes
        for value, data in children_data.items():
            child = self._build_tree(data['X'], data['y'], depth + 1, data['sample_weight'])
            
            # Add class level or condition information
            if 'class_level' in data:
                child.class_level = data['class_level']
            elif 'condition' in data:
                child.condition = data['condition']
                
            node.children[value] = child
        
        return node
    
    def fit(self, X, y, sample_weight=None):
        """Build the Parv decision tree from training data"""
        # Convert pandas DataFrame/Series to numpy arrays if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        # Validate input
        X, y = check_X_y(X, y, dtype=None)
        
        # Store number of features and classes
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Set up class extractor
        self.class_extractor = self._setup_class_extractor(X)
        
        # Set up class weights and importance
        self.class_weights_ = self._setup_class_weights(y)
        
        # Set up sample weights
        if sample_weight is None and self.class_weights_ is not None:
            sample_weight = np.array([self.class_weights_.get(cls, 1.0) for cls in y])
        
        # Set up class importance
        self.class_importance = {}
        if self.class_hierarchy is not None:
            for i, level in enumerate(self.class_hierarchy):
                # Higher values in the hierarchy are more important
                self.class_importance[level] = len(self.class_hierarchy) - i
        
        # Set random state
        self.random_state_ = check_random_state(self.random_state)
            
        # Build the tree
        self.tree_ = self._build_tree(X, y, sample_weight=sample_weight)
        
        # Clear computation cache
        self._computation_cache.clear()
        
        # Mark as fitted
        self.is_fitted_ = True
        
        return self
    
    def _predict_sample_proba(self, x, node=None):
        """Predict class probabilities for a single sample"""
        if node is None:
            node = self.tree_
            
        if node.leaf:
            # For leaf nodes, return the class distribution
            proba = np.zeros(len(self.classes_))
            total = sum(node.class_distribution.values())
            
            for i, cls in enumerate(self.classes_):
                proba[i] = node.class_distribution.get(cls, 0) / total
                
            return proba
            
        if node.feature_type == 'categorical':
            # Categorical feature
            value = x[node.feature_idx]
            
            if value in node.children:
                return self._predict_sample_proba(x, node.children[value])
            else:
                # If value not seen during training, use node's distribution
                proba = np.zeros(len(self.classes_))
                total = sum(node.class_distribution.values())
                
                for i, cls in enumerate(self.classes_):
                    proba[i] = node.class_distribution.get(cls, 0) / total
                    
                return proba
                
        else:
            # Continuous feature
            value = x[node.feature_idx]
            
            if value <= node.threshold and 'left' in node.children:
                return self._predict_sample_proba(x, node.children['left'])
            elif value > node.threshold and 'right' in node.children:
                return self._predict_sample_proba(x, node.children['right'])
            else:
                # If no matching child, use node's distribution
                proba = np.zeros(len(self.classes_))
                total = sum(node.class_distribution.values())
                
                for i, cls in enumerate(self.classes_):
                    proba[i] = node.class_distribution.get(cls, 0) / total
                    
                return proba
    
    def _predict_sample(self, x):
        """Predict class for a single sample"""
        proba = self._predict_sample_proba(x)
        return self.classes_[np.argmax(proba)]
    
    def predict(self, X):
        """Predict classes for samples in X"""
        # Check if fitted
        check_is_fitted(self, ['is_fitted_', 'tree_'])
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
            
        # Validate input
        X = check_array(X, dtype=None)
        
        # Ensure correct number of features
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but ParvClassifier "
                            f"is expecting {self.n_features_in_} features")
        
        # Predict each sample
        return np.array([self._predict_sample(x) for x in X])
    
    def predict_proba(self, X):
        """Predict class probabilities for X"""
        # Check if fitted
        check_is_fitted(self, ['is_fitted_', 'tree_'])
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
            
        # Validate input
        X = check_array(X, dtype=None)
        
        # Ensure correct number of features
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but ParvClassifier "
                            f"is expecting {self.n_features_in_} features")
        
        # Predict probabilities for each sample
        return np.array([self._predict_sample_proba(x) for x in X])
    
    def export_tree(self, feature_names=None, class_names=None):
        """Export the tree as a text representation"""
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Tree not fitted. Call fit before exporting tree.")
            
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features_in_)]
            
        if class_names is None:
            class_names = self.classes_
            
        def _export_node(node, indent=""):
            if node.leaf:
                class_idx = np.where(self.classes_ == node.prediction)[0][0]
                class_name = class_names[class_idx]
                
                # Show distribution for leaf nodes
                dist_str = ", ".join([f"{class_names[i]}:{node.class_distribution.get(cls, 0)}" 
                                    for i, cls in enumerate(self.classes_)])
                
                return f"{indent}→ {class_name} [samples={node.samples}, distribution={{{dist_str}}}]\n"
                
            feature_name = feature_names[node.feature_idx]
            
            if node.feature_type == 'categorical':
                # Categorical split
                output = f"{indent}{feature_name} [Hybrid Gain={node.hybrid_gain:.4f}, "
                output += f"Parv Gain={node.Parv_gain:.4f}, Samples={node.samples}]\n"
                
                for value, child in node.children.items():
                    # Show class level for categorical values
                    class_level = ""
                    if hasattr(child, 'class_level') and child.class_level:
                        class_level = f", Class={child.class_level}"
                        
                    output += f"{indent}|-- {value}{class_level}: {_export_node(child, indent + '|   ')}"
            else:
                # Continuous split
                output = f"{indent}{feature_name} <= {node.threshold:.4f} "
                output += f"[Hybrid Gain={node.hybrid_gain:.4f}, Parv Gain={node.Parv_gain:.4f}, "
                output += f"Samples={node.samples}]\n"
                
                if 'left' in node.children:
                    output += f"{indent}|-- <= {node.threshold:.4f}: {_export_node(node.children['left'], indent + '|   ')}"
                
                if 'right' in node.children:
                    output += f"{indent}|-- > {node.threshold:.4f}: {_export_node(node.children['right'], indent + '|   ')}"
                
            return output
            
        return _export_node(self.tree_)
    
class SequentialDecisionProcess:
    """
    Sequential Decision Process for tracking temporal changes in Parv/PARV values
    across multiple samples/time periods from the same locations.
    
    This class uses the ParvDecisionTreeClassifier to calculate Parv/PARV index
    for each sample/time period and tracks how these values change over time.
    
    Parameters:
    -----------
    base_classifier : ParvDecisionTreeClassifier
        The base classifier used to calculate Parv/PARV values
        
    location_column : str, default=None
        The column that identifies unique locations. If None, assumes
        the first column is the location identifier.
        
    time_column : str, default=None
        The column that identifies time periods. If None, assumes
        samples are already ordered by time.
        
    feature_columns : list, default=None
        The columns to use as features. If None, uses all columns
        except location_column, time_column, and label_column.
        
    label_column : str, default='Label'
        The column containing class labels.
        
    class_hierarchy : list, default=None
        Ordered list of class levels from highest to lowest importance.
    """
    
    def __init__(self, base_classifier, location_column=None, time_column=None,
                feature_columns=None, label_column='Label', class_hierarchy=None):
        self.base_classifier = base_classifier
        self.location_column = location_column
        self.time_column = time_column
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.class_hierarchy = class_hierarchy
        
        # Storage for temporal results
        self.temporal_Parv_values = {}
        self.temporal_class_level_Parv = {}
        self.location_shifts = {}
        self.fitted = False
        
    def _prepare_data_for_period(self, data, period):
        """Prepare data for a specific time period"""
        if self.time_column is not None:
            period_data = data[data[self.time_column] == period].copy()
        else:
            period_data = data.copy()
            
        # Extract features, labels, and location IDs
        if self.feature_columns is not None:
            X = period_data[self.feature_columns]
        else:
            # Use all columns except location, time, and label
            exclude_cols = [col for col in [self.location_column, self.time_column, self.label_column] if col is not None]
            X = period_data.drop(exclude_cols, axis=1)
            
        y = period_data[self.label_column]
        
        if self.location_column is not None:
            locations = period_data[self.location_column]
        else:
            # Use index as location if not specified
            locations = period_data.index
            
        return X, y, locations
    
    def fit(self, data):
        """
        Fit the sequential decision process to the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data to fit the model to. Should contain columns for
            locations, time periods (optional), features, and labels.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Identify time periods
        if self.time_column is not None:
            time_periods = sorted(data[self.time_column].unique())
        else:
            # If no time column, assume one time period
            time_periods = [0]
            
        # Process each time period
        for period in time_periods:
            X, y, locations = self._prepare_data_for_period(data, period)
            
            # Fit the base classifier
            self.base_classifier.fit(X, y)
            
            # Calculate Parv values
            Parv_values, class_level_Parv = self._calculate_Parv_values(X, y)
            
            # Store results for this time period
            self.temporal_Parv_values[period] = Parv_values
            self.temporal_class_level_Parv[period] = class_level_Parv
        
        # Calculate shifts between consecutive time periods
        if len(time_periods) > 1:
            self._calculate_temporal_shifts(time_periods)
            
        self.fitted = True
        return self
    
    def _calculate_Parv_values(self, X, y):
        """Calculate Parv values for features and class levels"""
        # Get feature Gini results from classifier
        feature_gini_results = {}
        for feature_idx in range(X.shape[1]):
            feature_gini_results[feature_idx] = self.base_classifier._calculate_feature_gini(
                X.values, y.values, feature_idx
            )
        
        # Calculate class level Parv
        class_level_Parv, total_Parv = self.base_classifier._calculate_Parv(
            X.values, y.values, feature_gini_results
        )
        
        # Create feature-level Parv dictionary
        feature_Parv = {}
        feature_names = X.columns
        
        for feature_idx, result in feature_gini_results.items():
            if result['is_categorical']:
                feature_name = feature_names[feature_idx]
                weighted_gini = result['weighted_gini']
                parent_gini = self.base_classifier._calculate_weighted_gini(y.values)
                
                # Calculate Parv for this feature
                feature_Parv[feature_name] = parent_gini - weighted_gini
        
        return feature_Parv, class_level_Parv
    
    def _calculate_temporal_shifts(self, time_periods):
        """Calculate shifts in Parv values between consecutive time periods"""
        for i in range(len(time_periods) - 1):
            current_period = time_periods[i]
            next_period = time_periods[i + 1]
            
            # Calculate shifts for overall Parv
            feature_shifts = {}
            for feature in self.temporal_Parv_values[current_period]:
                if feature in self.temporal_Parv_values[next_period]:
                    current_Parv = self.temporal_Parv_values[current_period][feature]
                    next_Parv = self.temporal_Parv_values[next_period][feature]
                    feature_shifts[feature] = next_Parv - current_Parv
            
            # Calculate shifts for class level Parv
            class_level_shifts = {}
            for class_level in self.temporal_class_level_Parv[current_period]:
                if class_level in self.temporal_class_level_Parv[next_period]:
                    current_Parv = self.temporal_class_level_Parv[current_period][class_level]
                    next_Parv = self.temporal_class_level_Parv[next_period][class_level]
                    class_level_shifts[class_level] = next_Parv - current_Parv
            
            # Store shifts for this transition
            transition_key = f"{current_period}_to_{next_period}"
            self.location_shifts[transition_key] = {
                'feature_shifts': feature_shifts,
                'class_level_shifts': class_level_shifts
            }
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the base classifier.
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            The input samples.
            
        Returns:
        --------
        probas : ndarray
            The class probabilities of the input samples.
        """
        return self.base_classifier.predict_proba(X)
    
    def predict(self, X):
        """
        Predict class labels using the base classifier.
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            The input samples.
            
        Returns:
        --------
        y_pred : ndarray
            The predicted class labels.
        """
        return self.base_classifier.predict(X)
    
    def get_Parv_values(self, time_period=None):
        """
        Get Parv values for a specific time period or all time periods.
        
        Parameters:
        -----------
        time_period : object, default=None
            The time period to get Parv values for. If None, returns
            Parv values for all time periods.
            
        Returns:
        --------
        Parv_values : dict
            Dictionary of Parv values by feature and time period.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit first.")
            
        if time_period is not None:
            if time_period not in self.temporal_Parv_values:
                raise ValueError(f"Time period {time_period} not found in fitted data.")
            return self.temporal_Parv_values[time_period]
        
        return self.temporal_Parv_values
    
    def get_class_level_Parv(self, time_period=None):
        """
        Get class level Parv values for a specific time period or all time periods.
        
        Parameters:
        -----------
        time_period : object, default=None
            The time period to get class level Parv values for. If None, returns
            class level Parv values for all time periods.
            
        Returns:
        --------
        class_level_Parv : dict
            Dictionary of class level Parv values by time period.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit first.")
            
        if time_period is not None:
            if time_period not in self.temporal_class_level_Parv:
                raise ValueError(f"Time period {time_period} not found in fitted data.")
            return self.temporal_class_level_Parv[time_period]
        
        return self.temporal_class_level_Parv
    
    def get_temporal_shifts(self, transition=None):
        """
        Get shifts in Parv values between consecutive time periods.
        
        Parameters:
        -----------
        transition : str, default=None
            The transition to get shifts for, in the format "period1_to_period2".
            If None, returns shifts for all transitions.
            
        Returns:
        --------
        shifts : dict
            Dictionary of shifts in Parv values by transition.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit first.")
            
        if transition is not None:
            if transition not in self.location_shifts:
                raise ValueError(f"Transition {transition} not found in fitted data.")
            return self.location_shifts[transition]
        
        return self.location_shifts
    
    def plot_temporal_Parv(self, class_level=None, feature=None, figsize=(12, 6)):
        """
        Plot temporal changes in Parv values.
        
        Parameters:
        -----------
        class_level : str, default=None
            If provided, plots Parv values for this class level over time.
            If None, plots feature-level Parv values.
            
        feature : str, default=None
            If provided and class_level is None, plots Parv values for this feature over time.
            If None and class_level is None, plots Parv values for all features.
            
        figsize : tuple, default=(12, 6)
            Figure size.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit first.")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Identify time periods
        time_periods = sorted(self.temporal_Parv_values.keys())
        
        if class_level is not None:
            # Plot class level Parv values over time
            Parv_values = [self.temporal_class_level_Parv[period].get(class_level, 0) 
                          for period in time_periods]
            
            ax.plot(time_periods, Parv_values, marker='o', linewidth=2, label=class_level)
            ax.set_title(f'Temporal Changes in Parv for Class Level: {class_level}')
            ax.set_ylabel('Parv Value')
            
        else:
            # Plot feature-level Parv values over time
            if feature is not None:
                # Plot Parv values for a single feature
                Parv_values = [self.temporal_Parv_values[period].get(feature, 0) 
                              for period in time_periods]
                
                ax.plot(time_periods, Parv_values, marker='o', linewidth=2, label=feature)
                ax.set_title(f'Temporal Changes in Parv for Feature: {feature}')
                ax.set_ylabel('Parv Value')
                
            else:
                # Plot Parv values for all features
                all_features = set()
                for period in time_periods:
                    all_features.update(self.temporal_Parv_values[period].keys())
                
                for feature in all_features:
                    Parv_values = [self.temporal_Parv_values[period].get(feature, 0) 
                                  for period in time_periods]
                    
                    ax.plot(time_periods, Parv_values, marker='o', linewidth=2, label=feature)
                
                ax.set_title('Temporal Changes in Parv by Feature')
                ax.set_ylabel('Parv Value')
                
        ax.set_xlabel('Time Period')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_temporal_shifts(self, figsize=(14, 8), feature_limit=10, class_level_limit=None):
        """
        Plot shifts in Parv values between consecutive time periods.
        
        Parameters:
        -----------
        figsize : tuple, default=(14, 8)
            Figure size.
            
        feature_limit : int, default=10
            Maximum number of features to include in the plot.
            If None, includes all features.
            
        class_level_limit : int, default=None
            Maximum number of class levels to include in the plot.
            If None, includes all class levels.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit first.")
            
        if not self.location_shifts:
            raise ValueError("No temporal shifts available. Need at least two time periods.")
            
        # Create subplots - one for features, one for class levels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Prepare data for feature shifts
        transitions = list(self.location_shifts.keys())
        all_features = set()
        for transition in transitions:
            all_features.update(self.location_shifts[transition]['feature_shifts'].keys())
        
        # Limit number of features if needed
        if feature_limit is not None and len(all_features) > feature_limit:
            # Get features with highest absolute shifts
            feature_max_shifts = {}
            for feature in all_features:
                max_shift = max([abs(self.location_shifts[t]['feature_shifts'].get(feature, 0)) 
                                for t in transitions])
                feature_max_shifts[feature] = max_shift
            
            all_features = sorted(all_features, 
                                 key=lambda f: feature_max_shifts[f], 
                                 reverse=True)[:feature_limit]
        
        # Plot feature shifts
        feature_data = []
        for transition in transitions:
            for feature in all_features:
                shift = self.location_shifts[transition]['feature_shifts'].get(feature, 0)
                feature_data.append({
                    'Transition': transition,
                    'Feature': feature,
                    'Shift': shift
                })
        
        feature_df = pd.DataFrame(feature_data)
        if not feature_df.empty:
            pivot_df = feature_df.pivot(index='Feature', columns='Transition', values='Shift')
            sns.heatmap(pivot_df, cmap='coolwarm', center=0, ax=ax1, annot=True, fmt='.2f')
            ax1.set_title('Parv Shifts by Feature')
        else:
            ax1.text(0.5, 0.5, 'No feature shifts available', 
                    horizontalalignment='center', verticalalignment='center')
        
        # Prepare data for class level shifts
        all_class_levels = set()
        for transition in transitions:
            all_class_levels.update(self.location_shifts[transition]['class_level_shifts'].keys())
        
        # Limit number of class levels if needed
        if class_level_limit is not None and len(all_class_levels) > class_level_limit:
            # Get class levels with highest absolute shifts
            class_level_max_shifts = {}
            for cl in all_class_levels:
                max_shift = max([abs(self.location_shifts[t]['class_level_shifts'].get(cl, 0)) 
                                for t in transitions])
                class_level_max_shifts[cl] = max_shift
            
            all_class_levels = sorted(all_class_levels, 
                                     key=lambda cl: class_level_max_shifts[cl], 
                                     reverse=True)[:class_level_limit]
        
        # Plot class level shifts
        class_level_data = []
        for transition in transitions:
            for cl in all_class_levels:
                shift = self.location_shifts[transition]['class_level_shifts'].get(cl, 0)
                class_level_data.append({
                    'Transition': transition,
                    'Class Level': cl,
                    'Shift': shift
                })
        
        class_level_df = pd.DataFrame(class_level_data)
        if not class_level_df.empty:
            pivot_df = class_level_df.pivot(index='Class Level', columns='Transition', values='Shift')
            sns.heatmap(pivot_df, cmap='coolwarm', center=0, ax=ax2, annot=True, fmt='.2f')
            ax2.set_title('Parv Shifts by Class Level')
        else:
            ax2.text(0.5, 0.5, 'No class level shifts available', 
                    horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_file=None):
        """
        Generate a comprehensive report of temporal Parv analysis.
        
        Parameters:
        -----------
        output_file : str, default=None
            If provided, writes the report to this file.
            If None, returns the report as a string.
            
        Returns:
        --------
        report : str
            The report as a string, if output_file is None.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit first.")
            
        # Start building the report
        report = []
        report.append("# Temporal Parv Analysis Report")
        report.append("\n## Overview")
        
        # Time periods info
        time_periods = sorted(self.temporal_Parv_values.keys())
        report.append(f"\nNumber of time periods analyzed: {len(time_periods)}")
        report.append(f"Time periods: {', '.join(map(str, time_periods))}")
        
        # Parv values by time period
        report.append("\n## Parv Values by Time Period")
        
        for period in time_periods:
            report.append(f"\n### Time Period: {period}")
            
            # Feature Parv values
            report.append("\n#### Feature Parv Values")
            feature_Parv = self.temporal_Parv_values[period]
            
            if feature_Parv:
                # Sort features by Parv value
                sorted_features = sorted(feature_Parv.items(), key=lambda x: x[1], reverse=True)
                
                report.append("\n| Feature | Parv Value |")
                report.append("| ------- | --------- |")
                
                for feature, Parv in sorted_features:
                    report.append(f"| {feature} | {Parv:.4f} |")
            else:
                report.append("\nNo feature Parv values available.")
            
            # Class level Parv values
            report.append("\n#### Class Level Parv Values")
            class_level_Parv = self.temporal_class_level_Parv[period]
            
            if class_level_Parv:
                # Sort class levels by Parv value
                sorted_cls = sorted(class_level_Parv.items(), key=lambda x: x[1], reverse=True)
                
                report.append("\n| Class Level | Parv Value |")
                report.append("| ----------- | --------- |")
                
                for cls, Parv in sorted_cls:
                    report.append(f"| {cls} | {Parv:.4f} |")
            else:
                report.append("\nNo class level Parv values available.")
        
        # Temporal shifts
        if self.location_shifts:
            report.append("\n## Temporal Shifts in Parv Values")
            
            for transition, shifts in self.location_shifts.items():
                report.append(f"\n### Transition: {transition}")
                
                # Feature shifts
                report.append("\n#### Feature Parv Shifts")
                feature_shifts = shifts['feature_shifts']
                
                if feature_shifts:
                    # Sort features by absolute shift value
                    sorted_features = sorted(feature_shifts.items(), 
                                            key=lambda x: abs(x[1]), 
                                            reverse=True)
                    
                    report.append("\n| Feature | Parv Shift |")
                    report.append("| ------- | --------- |")
                    
                    for feature, shift in sorted_features:
                        # Add + sign for positive shifts
                        shift_str = f"+{shift:.4f}" if shift > 0 else f"{shift:.4f}"
                        report.append(f"| {feature} | {shift_str} |")
                else:
                    report.append("\nNo feature Parv shifts available.")
                
                # Class level shifts
                report.append("\n#### Class Level Parv Shifts")
                class_level_shifts = shifts['class_level_shifts']
                
                if class_level_shifts:
                    # Sort class levels by absolute shift value
                    sorted_cls = sorted(class_level_shifts.items(), 
                                       key=lambda x: abs(x[1]), 
                                       reverse=True)
                    
                    report.append("\n| Class Level | Parv Shift |")
                    report.append("| ----------- | --------- |")
                    
                    for cls, shift in sorted_cls:
                        # Add + sign for positive shifts
                        shift_str = f"+{shift:.4f}" if shift > 0 else f"{shift:.4f}"
                        report.append(f"| {cls} | {shift_str} |")
                else:
                    report.append("\nNo class level Parv shifts available.")
        
        # Join the report
        report_str = "\n".join(report)
        
        # Write to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_str)
            return None
        
        return report_str
