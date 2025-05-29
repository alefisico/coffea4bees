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

# Set up module logger
logger = logging.getLogger('JCMTools')


class jetCombinatoricModel:
    """
    Main class for the Jet Combinatoric Model (JCM).
    
    The JCM is used to reweight the 3-tag multijet events to model 
    the 4-tag multijet background. The model fits the jet multiplicity
    distribution to determine the weights.
    
    Attributes:
        parameters (List[Dict[str, Any]]): List of all model parameters
        pseudoTagProb (Dict[str, Any]): Probability of a light jet to be tagged
        pairEnhancement (Dict[str, Any]): Enhancement for even number of tags
        pairEnhancementDecay (Dict[str, Any]): Decay parameter for pair enhancement
        threeTightTagFraction (Dict[str, Any]): Normalizing parameter
        tt4b_nTagJets (np.ndarray): Number of tagged jets in tt 4-tag events
        tt4b_nTagJets_errors (np.ndarray): Errors on tt 4-tag tagged jets
        qcd3b (np.ndarray): QCD 3-tag events
        qcd3b_errors (np.ndarray): Errors on QCD 3-tag events
        tt4b (np.ndarray): tt 4-tag events
        fit_parameters (List[Dict[str, Any]]): Parameters used in the fit (unfixed only)
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
        self.pseudoTagProb = {
            "name": "pseudoTagProb", "value": None, "error": None, "percentError": None,
            "index": 0, "lowerLimit": 0, "upperLimit": 1, "default": 0.05, "fix": None
        }
        self.pairEnhancement = {
            "name": "pairEnhancement", "value": None, "error": None, "percentError": None,
            "index": 1, "lowerLimit": 0, "upperLimit": 3, "default": 1.0, "fix": None
        }
        self.pairEnhancementDecay = {
            "name": "pairEnhancementDecay", "value": None, "error": None, "percentError": None,
            "index": 2, "lowerLimit": 0.1, "upperLimit": 100, "default": 0.7, "fix": None
        }
        self.threeTightTagFraction = {
            "name": "threeTightTagFraction", "value": None, "error": None, "percentError": None,
            "index": 3, "lowerLimit": 0, "upperLimit": 1000000, "default": 1000, "fix": None
        }

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
            self.parameters_lower_bounds.append(p["lowerLimit"])
            self.parameters_upper_bounds.append(p["upperLimit"])
            self.default_parameters.append(p["default"])

        self.nParameters = len(self.parameters)
        
        # These will be set during fitting
        self.fit_chi2 = None
        self.fit_ndf = None
        self.fit_prob = None
        self.fit_errs = None
        
        # Function to use in fitting - will be set when fixing parameters
        self.bkgd_func_njet_constrained = None

    def dump(self) -> None:
        """Print all parameter values and their status."""
        for parameter in self.parameters:
            if parameter["value"] is not None and parameter["error"] is not None:
                parameter["percentError"] = parameter["error"] / parameter["value"] * 100 if parameter["value"] else 0
                logger.info(f"{parameter['name']}: {parameter['value']:.6f} +/- {parameter['error']:.6f} ({parameter['percentError']:.2f}%)")
            elif parameter["value"] is not None:
                logger.info(f"{parameter['name']}: {parameter['value']:.6f} (Fixed)")
            else:
                logger.info(f"{parameter['name']}: Not yet fitted or fixed")

    def fixParameters(self, names: List[str], values: List[float]) -> None:
        """
        Fix parameters to specified values.
        
        Args:
            names: List of parameter names to fix
            values: List of values to fix the parameters to
        """
        for ip, p in enumerate(self.parameters):
            for _iname, _name in enumerate(names):
                if p["name"] == _name:
                    logger.info(f"Fixing {_name} to {values[_iname]}")
                    p["fix"] = values[_iname]
                    p["value"] = values[_iname] # Also set the value when fixing
                    p["error"] = 0 # Error is 0 for fixed parameters

    def _reset_fit_parameters(self) -> None:
        """Reset the fit parameters after fixing some parameters."""
        self.fit_parameters = []
        self.default_parameters = []
        self.parameters_lower_bounds = []
        self.parameters_upper_bounds = []

        for p in self.parameters:
            if p["fix"] is not None:
                continue
            self.fit_parameters.append(p)
            self.default_parameters.append(p["default"])
            self.parameters_lower_bounds.append(p["lowerLimit"])
            self.parameters_upper_bounds.append(p["upperLimit"])

    def fixParameter_combination(self, params_to_fix: Dict[str, float]) -> None:
        """
        Fix multiple parameters at once with specific values, and set up the constrained function.
        
        This replaces the previous fixParameter_norm, fixParameter_d_norm, and fixParameter_e_d_norm
        with a more general solution.
        
        Args:
            params_to_fix: Dictionary of parameter names and values to fix
                           e.g. {"threeTightTagFraction": 0.5, "pairEnhancement": 0.0}
        """
        # Extract names and values for fixParameters
        names = list(params_to_fix.keys())
        values = list(params_to_fix.values())
        
        # Fix the specified parameters
        self.fixParameters(names, values)
        
        # Reset fit parameters
        self._reset_fit_parameters()
        
        # Determine which parameters are still free
        free_params = [p for p in self.parameters if p["fix"] is None]
        free_param_indices = [p["index"] for p in free_params]
        
        # Set up the background function with fixed parameters
        # Start with default values for all parameters
        f, e, d, norm = 0.05, 0.0, 1.0, 1.0
        
        # Update with fixed values from params_to_fix
        if "pseudoTagProb" in params_to_fix:
            f = params_to_fix["pseudoTagProb"]
        if "pairEnhancement" in params_to_fix:
            e = params_to_fix["pairEnhancement"]
        if "pairEnhancementDecay" in params_to_fix:
            d = params_to_fix["pairEnhancementDecay"]
        if "threeTightTagFraction" in params_to_fix:
            norm = params_to_fix["threeTightTagFraction"]
        
        # Create the appropriate lambda function based on which parameters are free
        # This approach dynamically creates the correct function signature
        if len(free_params) == 1 and free_params[0]["name"] == "pseudoTagProb":
            # Only f is free
            self.bkgd_func_njet_constrained = lambda x, f_val, debug=False: self.bkgd_func_njet(x, f_val, e, d, norm, debug)
        elif len(free_params) == 2 and 0 in free_param_indices and 1 in free_param_indices:
            # f and e are free
            self.bkgd_func_njet_constrained = lambda x, f_val, e_val, debug=False: self.bkgd_func_njet(x, f_val, e_val, d, norm, debug)
        elif len(free_params) == 3 and 0 in free_param_indices and 1 in free_param_indices and 2 in free_param_indices:
            # f, e, and d are free
            self.bkgd_func_njet_constrained = lambda x, f_val, e_val, d_val, debug=False: self.bkgd_func_njet(x, f_val, e_val, d_val, norm, debug)
        else:
            # Custom case - build a function that maps free parameters to their correct positions
            def create_constrained_func():
                def constrained_func(x, *args, debug=False):
                    # Create a full parameter list with fixed values
                    full_params = [f, e, d, norm]
                    
                    # Replace with the free parameters in the correct positions
                    for i, param in enumerate(free_params):
                        full_params[param["index"]] = args[i]
                    
                    # Unpack the parameters
                    return self.bkgd_func_njet(x, *full_params, debug=debug)
                
                return constrained_func
            
            self.bkgd_func_njet_constrained = create_constrained_func()
            
        logger.info(f"Fixed parameters: {', '.join([f'{n}={v}' for n, v in params_to_fix.items()])}")
        logger.info(f"Free parameters: {', '.join([p['name'] for p in free_params])}")


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
        nTags_pred_result = self.nTagPred(nTags, [f, e, d, norm])["values"]
        output[0:4] = nTags_pred_result[4:8]
        
        if debug:
            logger.debug(f"bkgd_func_njet output initial: {output}")

        # Add jet multiplicity component
        for ibin, this_nj in enumerate(nj):
            if this_nj < 4:
                continue

            w = np.sum(self.getPseudoTagProbs(this_nj, f, e, d, norm)[1:])
            output[this_nj] += w * self.qcd3b[this_nj] + self.tt4b[this_nj]

        if debug:
            logger.debug(f"bkgd_func_njet output final: {output}")
            
        return output

    def getPseudoTagProbs(self, nj: int, f: float, e: float = 0.0, d: float = 1.0, 
                        norm: float = 1.0) -> np.ndarray:
        """
        Calculate the pseudo-tag probabilities for a given jet multiplicity.
        
        Args:
            nj: Number of jets
            f: Pseudo-tag probability 
            e: Pair enhancement factor
            d: Pair enhancement decay parameter
            norm: Normalization factor
            
        Returns:
            Array of probabilities for each number of pseudo-tags
        """
        nbt = 3  # Number of required b-tags
        nlt = nj - nbt  # Number of selected untagged jets ("light" jets)
        nPseudoTagProb = np.zeros(nlt + 1)

        for npt in range(0, nlt + 1):   # npt is the number of pseudoTags in this combination
            nt = nbt + npt
            nnt = nlt - npt  # Number of not tagged

            # (ways to choose npt pseudoTags from nlt light jets) * pseudoTagProb^nlt * (1-pseudoTagProb)^{nlt-npt}
            w_npt = norm * comb(nlt, npt, exact=True) * f**npt * (1 - f)**nnt

            # Apply pair enhancement for even number of tags
            if (nt % 2) == 0:
                w_npt *= 1 + e / nlt**d

            logger.debug(f"npt: {npt}, w_npt: {w_npt}, nt: {nt}, nlt: {nlt}")
            nPseudoTagProb[npt] += w_npt
            
        return nPseudoTagProb

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
            try:
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
            except Exception as e:
                logger.error(f"Minimization failed: {e}")
                raise ValueError(f"Fit failed: {str(e)}")
        else:
            # Use curve_fit which provides the covariance matrix directly
            try:
                popt, errs = curve_fit(
                    self.bkgd_func_njet_constrained,
                    bin_centers,
                    bin_values,
                    self.default_parameters,
                    sigma=bin_errors,
                    bounds=(self.parameters_lower_bounds, self.parameters_upper_bounds),
                )
            except Exception as e:
                logger.error(f"Curve fit failed: {e}")
                raise ValueError(f"Fit failed: {str(e)}")
            
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
            if parameter["fix"] is not None:
                parameter["value"] = parameter["fix"]
                parameter["error"] = 0
                continue

            idx = parameter["index"]
            if idx < len(popt):
                parameter["value"] = popt[idx]
                parameter["error"] = sigma_p1[idx]
                # Update fit parameters for calls to other functions
                for i, param in enumerate(self.fit_parameters):
                    if param["index"] == idx:
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
            
        param_values = [p["value"] for p in self.fit_parameters]
        if None in param_values:
            raise ValueError("One or more parameters have no value. Run fit() first.")
            
        return self.bkgd_func_njet_constrained(n, *param_values)
    
    def nTagPred(self, n: np.ndarray, par: Optional[List[float]] = None) -> np.ndarray:
        """
        Get predicted values for the number of tagged jets using current fit parameters.
        
        Args:
            n: Array of number of tags
            par: Optional parameter values
            
        Returns:
            Dictionary of arrays of predicted values and errors
        """
        if par is None:
            param_values = [p["value"] for p in self.fit_parameters if p["value"] is not None]
            fixed_val = [self.threeTightTagFraction["fix"]] if self.threeTightTagFraction["fix"] is not None else []
            par = param_values + fixed_val
            logger.info(f"Using parameters: {par}")

        output = np.zeros(len(n))
        output = copy(self.tt4b_nTagJets)

        for ibin, this_nTag in enumerate(n):
            for nj in range(this_nTag, 14):
                # Select the right calculation mode based on number of parameters
                if len(par) == 3:
                    # [f, e, norm] - d fixed to 1.0
                    nPseudoTagProb = self.getPseudoTagProbs(nj, par[0], par[1], 1.0, par[2])
                elif len(par) == 2:
                    # [f, norm] - e fixed to 0.0, d fixed to 1.0
                    nPseudoTagProb = self.getPseudoTagProbs(nj, par[0], 0.0, 1.0, par[1])
                elif len(par) == 1:
                    # [f] - e fixed to 0.0, d fixed to 1.0, norm from instance
                    norm_value = self.threeTightTagFraction["fix"] if self.threeTightTagFraction["fix"] is not None else 1.0
                    nPseudoTagProb = self.getPseudoTagProbs(nj, par[0], 0.0, 1.0, norm_value)
                else:
                    # Full parameter set [f, e, d, norm]
                    nPseudoTagProb = self.getPseudoTagProbs(nj, par[0], par[1], par[2], par[3])
                
                logger.debug(f"nj: {nj}, this_nTag: {this_nTag}, nPseudoTagProb: {nPseudoTagProb}")
                output[ibin + 4] += nPseudoTagProb[this_nTag - 3] * self.qcd3b[nj]
                output[3] += nPseudoTagProb[0] * self.qcd3b[nj]
        
        logger.debug(f"output: {output}")
        return { "values": np.array(output, float), "errors": np.array(output**0.5, float) }

    def getCombinatoricWeightList(self) -> List[float]:
        """
        Get the list of combinatoric weights for jet multiplicities 4-15.
        
        Returns:
            List of weights to apply to 3-tag events for each jet multiplicity
        """
        output_weights, zerotag_output_weights = [], []
        
        # Verify parameters have been set
        param_values = [p["value"] for p in self.fit_parameters]
        if None in param_values:
            raise ValueError("One or more parameters have no value. Run fit() first.")
            
        fixed_val = [self.threeTightTagFraction["fix"]] if self.threeTightTagFraction["fix"] is not None else []
        params = param_values + fixed_val
        
        # Calculate weights for jet multiplicity 4 through 15
        for nj in range(4, 16):
            nj_pseudoTagProbs = self.getPseudoTagProbs(nj, *params)
            zerotag_output_weights.append( nj_pseudoTagProbs[0] )
            output_weights.append( np.sum(nj_pseudoTagProbs[1:]) )
        logger.info(f"Output weights: {output_weights}")

        return output_weights, zerotag_output_weights

