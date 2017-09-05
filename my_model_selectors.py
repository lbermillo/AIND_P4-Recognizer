import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    
    L is the likelihood of the fitted model
    p is the number of parameters
    N is the number of data points.
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        lowest_score = float('inf')
        best_model = None
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                n_features = self.X.shape[1]
                
                model = self.base_model(n_components)
                # L is the likelihood of the fitted model
                logL = model.score(self.X, self.lengths)
                # N is the number of data points.
                logN = np.log(len(self.X))
                
                # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/17
                # According to the formula, p(number of free parameters) is sum of these 4 terms:
                    # Transition probs are the transmat array- n* n
                    # Starting probabilities size n_components, but since they add up to 1.0, so it will be n - 1
                    # Number of means= n_components * n_features; Variances are the size of the covars array, s
                    #    since we are using "diag" so it will be n_components * n_features
                # So we will get the formula as n_components * n_components + 2 * n_components * n_features -1
                p = np.power(n_components, 2) + 2 * n_components * n_features - 1
                # compute BIC
                BIC_score = -2 * logL + p * logN
                # assign BIC_score to lowest_score if it is less than the lowest_score
                lowest_score = min(BIC_score, lowest_score)
                # assign current model to best_model if the lowest_score is the current BIC_score
                best_model = model if lowest_score == BIC_score else best_model
            
            except: 
                continue
                
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # Initialize variables
        best_score = float('-inf')
        best_model = None
        
        # Iterate through each component
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            # Get the list of words
            word_list = self.words.keys()
            
            try:
                scores = []
                model = self.base_model(n_components)
                # Iterate through each word in the list
                for word in word_list:
                    # score all words that is not this_word
                    if word == self.this_word: 
                        continue
                    # if word != self.this_word:
                    X_word, lengths_word = self.hwords[word]
                    scores.append(model.score(X_word, lengths_word))
                # calculate score
                current_score = model.score(self.X, self.lengths) - np.average(scores)
                # update best_score and best_model
                best_score = max(best_score, current_score)
                best_model = model if best_score == current_score else best_model
            except:
                pass
            
        return best_model
            

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
                
        # Initialize variables
        best_score = float('-inf')
        best_model = None
        score_sum = 0

        
        # By using the fit method during training, the Baum-Welch Expectation-Maximization (EM) algorithm is 
        # invoked iteratively to find the best estimate for the model for the number of hidden states 
        # specified from a group of sample sequences.
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                split_method = KFold(n_splits=min(3, len(self.sequences)))
                logL_list = []
                word_sequences = self.sequences
                # Break the training set into "folds" and rotate which fold is left out of training
                for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                    # In order to run hmmlearn training using the X,lengths tuples on the new folds, 
                    # subsets must be combined based on the indices given for the folds. 
                    # A helper utility has been provided in the asl_utils module named combine_sequences for this purpose.
                    X_train, lengths_train = combine_sequences(cv_train_idx, word_sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, word_sequences)
                    
                    # we train a single word using Gaussian hidden Markov models (HMM). 
                    model = GaussianHMM(n_components=n_components, n_iter=1000).fit(X_train, lengths_train)
                    # The "left out" fold scored
                    logL = model.score(X_test, lengths_test)
                    logL_list.append(logL)
                
                # compute average of logL and assign to best_score if it has a larger value than best_score
                avg_score = np.average(logL_list)
                best_score = max(avg_score, best_score)
                # assign best_model to the current model if the current avg_score is the best_score
                best_model = model if best_score == avg_score else best_model
                    
            except:
                continue
            
        return best_model