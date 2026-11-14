/**
 * Linear models: linear, ridge, lasso, elastic-net and logistic regression. scikit-learn's `linear_model`.
 * @module
 */
export { LinearRegression, type LinearRegressionParams } from './linear-regression.js';
export { LogisticRegression, type LogisticRegressionParams } from './logistic-regression.js';
export { ElasticNet, Lasso, Ridge, type ElasticNetParams, type LassoParams, type RidgeParams } from './penalized.js';
