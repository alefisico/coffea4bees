#!/usr/bin/env python3
"""
Jet Combinatoric Model (JCM) Tools

This module provides the core functionality for the Jet Combinatoric Model
used in HH→4b analysis to model the combinatorial background from 3-tag events.
It contains the model parameter and fitting classes, along with helper functions
for data manipulation and model evaluation.

Author: Coffea4bees team
"""

from scipy.special import comb
import numpy as np
from coffea.util import load
import logging
from base_class.plots.helpers import get_cut_dict
import base_class.plots.iPlot_config as cfg
import hist
from copy import copy, deepcopy
from scipy.optimize import curve_fit, minimize
import scipy.stats
from typing import List, Tuple, Dict, Optional, Union, Any
from base_class.jcm_tools.helpers import getPseudoTagProbs, getCombinatoricWeight

# Set up module logger
logger = logging.getLogger('JCMTools')


class modelParameter:
    """
    A class to represent a parameter in the Jet Combinatoric Model.
    
    Attributes:
        name (str): Parameter name
        value (float): Current parameter value (set after fitting)
        error (float): Error on the parameter value (set after fitting)
        percentError (float): Percent error on the parameter value
        index (int): Index of the parameter in the model
        lowerLimit (float): Lower bound for the parameter in fitting
        upperLimit (float): Upper bound for the parameter in fitting
        default (float): Default value for the parameter
        fix (float or None): Fixed value for the parameter, or None if not fixed
    """
    
    def __init__(self, name: str = "", index: int = 0, 
                 lowerLimit: float = 0, upperLimit: float = 1, 
                 default: float = 0.5, fix: Optional[float] = None):
        """
        Initialize a model parameter.
        
        Args:
            name: Name of the parameter
            index: Index in the parameter list
            lowerLimit: Lower bound for fitting
            upperLimit: Upper bound for fitting
            default: Default value
            fix: Fixed value (if None, parameter will be fitted)
        """
        self.name = name
        self.value = None
        self.error = None
        self.percentError = None
        self.index = index
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.default = default
        self.fix = fix

    def dump(self) -> None:
        """Print the parameter value, error, and percent error."""
        self.percentError = self.error / self.value * 100 if self.value else 0
        logger.info((f"{self.name}: {self.value:.6f} +/- {self.error:.6f} ({self.percentError:.2f}%)"))


class jetCombinatoricModel:
    """
    Main class for the Jet Combinatoric Model (JCM).
    
    The JCM is used to reweight the 3-tag multijet events to model 
    the 4-tag multijet background. The model fits the jet multiplicity
    distribution to determine the weights.
    
    Attributes:
        parameters (List[modelParameter]): List of all model parameters
        pseudoTagProb (modelParameter): Probability of a light jet to be tagged
        pairEnhancement (modelParameter): Enhancement for even number of tags
        pairEnhancementDecay (modelParameter): Decay parameter for pair enhancement
        threeTightTagFraction (modelParameter): Normalizing parameter
        tt4b_nTagJets (np.ndarray): Number of tagged jets in tt 4-tag events
        tt4b_nTagJets_errors (np.ndarray): Errors on tt 4-tag tagged jets
        qcd3b (np.ndarray): QCD 3-tag events
        qcd3b_errors (np.ndarray): Errors on QCD 3-tag events
        tt4b (np.ndarray): tt 4-tag events
        fit_parameters (List[float]): Parameters used in the fit (unfixed only)
        default_parameters (List[float]): Default values for fit parameters
        parameters_lower_bounds (List[float]): Lower bounds for fit parameters
        parameters_upper_bounds (List[float]): Upper bounds for fit parameters
        nParameters (int): Number of parameters in the model
        fit_chi2 (float): Chi-squared of the fit
        fit_ndf (int): Number of degrees of freedom in the fit
        fit_prob (float): P-value of the fit
        fit_errs (np.ndarray): Error matrix from the fit
    """

    def __init__(self, *, tt4b_nTagJets: np.ndarray, tt4b_nTagJets_errors: np.ndarray, 
                 qcd3b: np.ndarray, qcd3b_errors: np.ndarray, tt4b: np.ndarray):
        """
        Initialize the JCM model.
        
        Args:
            tt4b_nTagJets: Number of tagged jets in tt 4-tag events
            tt4b_nTagJets_errors: Errors on tt 4-tag tagged jets
            qcd3b: QCD 3-tag events 
            qcd3b_errors: Errors on QCD 3-tag events
            tt4b: tt 4-tag events
        """
        # Initialize model parameters with reasonable defaults and bounds
        self.pseudoTagProb = modelParameter(
            "pseudoTagProb", 
            index=0, 
            lowerLimit=0, 
            upperLimit=1, 
            default=0.05
        )
        self.pairEnhancement = modelParameter(
            "pairEnhancement", 
            index=1, 
            lowerLimit=0, 
            upperLimit=3, 
            default=1.0
        )
        self.pairEnhancementDecay = modelParameter(
            "pairEnhancementDecay", 
            index=2, 
            lowerLimit=0.1, 
            upperLimit=100, 
            default=0.7
        )
        self.threeTightTagFraction = modelParameter(
            "threeTightTagFraction", 
            index=3, 
            lowerLimit=0, 
            upperLimit=1000000, 
            default=1000
        )

        self.parameters = [
            self.pseudoTagProb, 
            self.pairEnhancement, 
            self.pairEnhancementDecay, 
            self.threeTightTagFraction
        ]

        # Store input data
        self.tt4b_nTagJets = tt4b_nTagJets
        self.tt4b_nTagJets_errors = tt4b_nTagJets_errors
        self.qcd3b = qcd3b
        self.qcd3b_errors = qcd3b_errors
        self.tt4b = tt4b

        # Setup fit parameters
        self.default_parameters = []
        self.fit_parameters = []
        self.parameters_lower_bounds = []
        self.parameters_upper_bounds = []
        
        for p in self.parameters:
            self.fit_parameters.append(p)
            self.parameters_lower_bounds.append(p.lowerLimit)
            self.parameters_upper_bounds.append(p.upperLimit)
            self.default_parameters.append(p.default)

        self.nParameters = len(self.parameters)
        
        # These will be set during fitting
        self.fit_chi2 = None
        self.fit_ndf = None
        self.fit_prob = None
        self.fit_errs = None
        
        # Function to use in fitting - will be set when fixing parameters
        self.bkgd_func_njet_constrained = None

    def dump(self) -> None:
        """Print all parameter values."""
        for parameter in self.parameters:
            parameter.dump()

    def fixParameters(self, names: List[str], values: List[float]) -> None:
        """
        Fix parameters to specified values.
        
        Args:
            names: List of parameter names to fix
            values: List of values to fix the parameters to
        """
        for ip, p in enumerate(self.parameters):
            for _iname, _name in enumerate(names):
                if p.name == _name:
                    logger.info(f"Fixing {_name} to {values[_iname]}")
                    p.fix = values[_iname]

    def fixParameter_norm(self, value: float) -> None:
        """
        Fix the threeTightTagFraction normalization parameter.
        
        Args:
            value: Value to fix the parameter to
        """
        self.fixParameters(["threeTightTagFraction"], [value])
        
        # Reset fit parameters
        self._reset_fit_parameters()

        # Fix the normalization to the threeTightTagFraction
        self.bkgd_func_njet_constrained = lambda x, f, e, d, debug=False: self.bkgd_func_njet(x, f, e, d, value, debug)

    def fixParameter_d_norm(self, value: float) -> None:
        """
        Fix both the pairEnhancementDecay and threeTightTagFraction parameters.
        
        Args:
            value: Value to fix the threeTightTagFraction parameter to
            (pairEnhancementDecay is always fixed to 1.0)
        """
        self.fixParameters(["threeTightTagFraction", "pairEnhancementDecay"], [value, 1.0])
        
        # Reset fit parameters
        self._reset_fit_parameters()

        # Fix both parameters
        self.bkgd_func_njet_constrained = lambda x, f, e, debug=False: self.bkgd_func_njet(x, f, e, 1.0, value, debug)

    def fixParameter_e_d_norm(self, value: float) -> None:
        """
        Fix pairEnhancement, pairEnhancementDecay, and threeTightTagFraction parameters.
        
        Args:
            value: Value to fix the threeTightTagFraction parameter to
            (pairEnhancement is fixed to 0.0, pairEnhancementDecay to 1.0)
        """
        self.fixParameters(
            ["threeTightTagFraction", "pairEnhancement", "pairEnhancementDecay"], 
            [value, 0.0, 1.0]
        )
        
        # Reset fit parameters
        self._reset_fit_parameters()

        # Fix all three parameters
        self.bkgd_func_njet_constrained = lambda x, f, debug=False: self.bkgd_func_njet(x, f, 0.0, 1.0, value, debug)

    def _reset_fit_parameters(self) -> None:
        """Reset the fit parameters after fixing some parameters."""
        self.fit_parameters = []
        self.default_parameters = []
        self.parameters_lower_bounds = []
        self.parameters_upper_bounds = []

        for p in self.parameters:
            if p.fix is not None:
                continue
            self.fit_parameters.append(p)
            self.default_parameters.append(p.default)
            self.parameters_lower_bounds.append(p.lowerLimit)
            self.parameters_upper_bounds.append(p.upperLimit)

    def _nTagPred_values(self, par: List[float], n: np.ndarray) -> np.ndarray:
        """
        Calculate predicted values for the number of tagged jets.
        
        Args:
            par: List of parameters [f, e, d, norm]
            n: Array of number of tags
            
        Returns:
            Array of predicted values
        """
        output = np.zeros(len(n))
        output = copy(self.tt4b_nTagJets)

        for ibin, this_nTag in enumerate(n):
            for nj in range(this_nTag, 14):
                if len(par) == 3:
                    # [f, e, norm] - d fixed to 1.0
                    nPseudoTagProb = getPseudoTagProbs(nj, par[0], par[1], 1.0, par[2])
                elif len(par) == 2:
                    # [f, norm] - e fixed to 0.0, d fixed to 1.0
                    nPseudoTagProb = getPseudoTagProbs(nj, par[0], 0.0, 1.0, par[1])
                else:
                    # Full parameter set [f, e, d, norm]
                    nPseudoTagProb = getPseudoTagProbs(nj, par[0], par[1], par[2], par[3])
                
                output[ibin + 4] += nPseudoTagProb[this_nTag - 3] * self.qcd3b[nj]

        return np.array(output, float)

    def nTagPred_values(self, n: np.ndarray) -> np.ndarray:
        """
        Get predicted values for the number of tagged jets using current fit parameters.
        
        Args:
            n: Array of number of tags
            
        Returns:
            Array of predicted values
        """
        # Extract parameter values instead of objects
        param_values = [p.value for p in self.fit_parameters]
        fixed_val = [self.threeTightTagFraction.fix] if self.threeTightTagFraction.fix is not None else []
        
        return self._nTagPred_values(param_values + fixed_val, n)

    def nJetPred_values(self, n: np.ndarray) -> np.ndarray:
        """
        Get predicted values for jet multiplicity using current fit parameters.
        
        Args:
            n: Array of jet multiplicities
            
        Returns:
            Array of predicted values
        """
        if self.bkgd_func_njet_constrained is None:
            raise ValueError("Constrained function is not set. Call fixParameter_* first.")
        return self.bkgd_func_njet_constrained(n, *[p.value for p in self.fit_parameters])

    def getCombinatoricWeightList(self) -> List[float]:
        """
        Get the list of combinatoric weights for jet multiplicities 4-15.
        
        Returns:
            List of weights to apply to 3-tag events for each jet multiplicity
        """
        output_weights = []
        fixed_val = [self.threeTightTagFraction.fix] if self.threeTightTagFraction.fix is not None else []
        
        # Calculate weights for jet multiplicity 4 through 15
        for nj in range(4, 16):
            param_values = [p.value for p in self.fit_parameters] + fixed_val
            output_weights.append(getCombinatoricWeight(nj, *param_values))
            
        return output_weights

    def _nTagPred_errors(self, par: List[float], n: np.ndarray) -> np.ndarray:
        """
        Calculate errors on predicted values for the number of tagged jets.
        
        Args:
            par: List of parameters [f, e, d, norm]
            n: Array of number of tags
            
        Returns:
            Array of prediction errors
        """
        output = np.zeros(len(n))
        output = self.tt4b_nTagJets_errors**2

        for ibin, this_nTag in enumerate(n):
            for nj in range(this_nTag, 14):
                if len(par) == 3:
                    # [f, e, norm] - d fixed to 1.0
                    nPseudoTagProb = getPseudoTagProbs(nj, par[0], par[1], 1.0, par[2])
                elif len(par) == 2:
                    # [f, norm] - e fixed to 0.0, d fixed to 1.0
                    nPseudoTagProb = getPseudoTagProbs(nj, par[0], 0.0, 1.0, par[1])
                else:
                    # Full parameter set [f, e, d, norm]
                    nPseudoTagProb = getPseudoTagProbs(nj, par[0], par[1], par[2], par[3])

                output[ibin + 4] += (nPseudoTagProb[this_nTag - 3] * self.qcd3b_errors[nj])**2

        # Take the square root to get errors
        output = output**0.5
        return np.array(output, float)

    def nTagPred_errors(self, n: np.ndarray) -> np.ndarray:
        """
        Get errors on predicted values for the number of tagged jets.
        
        Args:
            n: Array of number of tags
            
        Returns:
            Array of prediction errors
        """
        # Extract parameter values instead of objects
        param_values = [p.value for p in self.fit_parameters]
        fixed_val = [self.threeTightTagFraction.fix] if self.threeTightTagFraction.fix is not None else []
        
        return self._nTagPred_errors(param_values + fixed_val, n)

    def bkgd_func_njet(self, x: np.ndarray, f: float, e: float, d: float, 
                       norm: float, debug: bool = False) -> np.ndarray:
        """
        Background model function for jet multiplicity.
        
        Args:
            x: Jet multiplicity bins
            f: pseudoTagProb parameter
            e: pairEnhancement parameter
            d: pairEnhancementDecay parameter
            norm: threeTightTagFraction parameter
            debug: Whether to print debug information
            
        Returns:
            Predicted values for each bin
        """
        nj = x.astype(int)
        output = np.zeros(len(x))

        # Add the n-tag component
        nTags = nj + 4
        nTags_pred_result = self._nTagPred_values([f, e, d, norm], nTags)
        output[0:4] = nTags_pred_result[4:8]
        
        if debug:
            logger.debug(f"bkgd_func_njet output initial: {output}")

        # Add jet multiplicity component
        for ibin, this_nj in enumerate(nj):
            if this_nj < 4:
                continue

            w = getCombinatoricWeight(this_nj, f, e, d, norm)
            output[this_nj] += w * self.qcd3b[this_nj]
            output[this_nj] += self.tt4b[this_nj]

        if debug:
            logger.debug(f"bkgd_func_njet output final: {output}")
            
        return output

    def fit(self, bin_centers: np.ndarray, bin_values: np.ndarray, 
            bin_errors: np.ndarray, scipy_optimize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the fit of the JCM model to data.
        
        Args:
            bin_centers: Bin centers (jet multiplicities)
            bin_values: Bin values (event counts)
            bin_errors: Bin errors
            scipy_optimize: Whether to use scipy.optimize.minimize instead of curve_fit
            
        Returns:
            Tuple of (residuals, pulls)
        """
        if self.bkgd_func_njet_constrained is None:
            raise ValueError("Constrained function is not set. Call fixParameter_* first.")
            
        logger.info(f"Fitting with {len(self.fit_parameters)} free parameters")
            
        # Do the fit
        if scipy_optimize:
            # Define the objective function (sum of squared residuals)
            def objective_function(params):
                model_values = self.bkgd_func_njet_constrained(bin_centers, *params)
                residuals = (bin_values - model_values) / bin_errors
                return np.sum(residuals**2)
            
            # Perform the minimization
            result = minimize(
                objective_function,
                self.default_parameters,
                bounds=list(zip(self.parameters_lower_bounds, self.parameters_upper_bounds)),
                method='L-BFGS-B',  # Change to another minimizer if needed
                options={'maxiter': 5000}
            )
            
            # Extract the optimized parameters
            popt = result.x
            
            # Extract the covariance matrix and compute errors
            if hasattr(result, 'hess_inv'):
                try:
                    if hasattr(result.hess_inv, 'todense'):
                        errs = np.array(result.hess_inv.todense())
                    else:
                        errs = np.array(result.hess_inv)
                except Exception as e:
                    logger.warning(f"Error converting Hessian: {e}")
                    errs = np.eye(len(popt)) * 0.001  # Fallback
            else:
                errs = np.eye(len(popt)) * 0.001  # Fallback
                logger.warning("Hessian not available, using default errors")
        else:
            # Use curve_fit which provides the covariance matrix directly
            popt, errs = curve_fit(
                self.bkgd_func_njet_constrained,
                bin_centers,
                bin_values,
                self.default_parameters,
                sigma=bin_errors,
                bounds=(self.parameters_lower_bounds, self.parameters_upper_bounds),
                # maxfev=10000,  # Increase max function evaluations
                # method='trf',  # Use Trust Region Reflective algorithm
                # absolute_sigma=True,  # Use absolute values of sigma for error calculation
            )
            
        # Store the fit error matrix
        self.fit_errs = errs
        
        # Calculate parameter errors from the covariance matrix diagonal
        sigma_p1 = []
        for i in range(len(popt)):
            try:
                sigma_p1.append(np.sqrt(np.abs(errs[i][i])))
            except (IndexError, ValueError) as e:
                logger.warning(f"Error calculating parameter error: {e}")
                sigma_p1.append(0.001)  # Default error

        # Update parameter values and errors
        for ip, parameter in enumerate(self.parameters):
            if parameter.fix is not None:
                parameter.value = parameter.fix
                parameter.error = 0
                continue

            idx = parameter.index
            if idx < len(popt):
                parameter.value = popt[idx]
                parameter.error = sigma_p1[idx]
                # Update fit parameters for calls to other functions
                for i, param in enumerate(self.fit_parameters):
                    if param.index == idx:
                        self.fit_parameters[i] = parameter
                        break

        # Calculate fit quality metrics
        self.fit_chi2 = np.sum(
            (self.bkgd_func_njet_constrained(bin_centers, *popt) - bin_values)**2 / bin_errors**2
        )
        self.fit_ndf = len(bin_values) - len(popt)
        self.fit_prob = scipy.stats.chi2.sf(self.fit_chi2, self.fit_ndf)

        # Calculate residuals and pulls
        residuals = bin_values - self.bkgd_func_njet_constrained(bin_centers, *popt)
        pulls = residuals / bin_errors
        
        logger.info(f"Fit completed: chi^2/ndf = {self.fit_chi2:.2f}/{self.fit_ndf} = " +
                   f"{self.fit_chi2/self.fit_ndf:.2f}, p-value = {self.fit_prob:.6f}")
        
        return residuals, pulls
