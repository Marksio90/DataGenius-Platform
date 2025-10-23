import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  TextField,
  Tabs,
  Tab,
  Alert,
  CircularProgress,
  Chip
} from '@mui/material';
import {
  Psychology,
  Speed,
  Timeline,
  BugReport,
  AutoAwesome,
  Insights
} from '@mui/icons-material';
import axios from 'axios';

function TabPanel({ children, value, index }) {
  return (
    <div hidden={value !== index}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export default function AdvancedFeatures({ sessionId }) {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Neural Networks
  const [nnFramework, setNnFramework] = useState('pytorch');
  const [nnAutoTune, setNnAutoTune] = useState(false);

  // Optimization
  const [optimizer, setOptimizer] = useState('optuna');
  const [nTrials, setNTrials] = useState(50);

  // Time Series
  const [tsMethod, setTsMethod] = useState('prophet');
  const [forecastPeriods, setForecastPeriods] = useState(30);

  // Explainability
  const [explainMethod, setExplainMethod] = useState('shap');
  const [sampleIndex, setSampleIndex] = useState(0);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    setResult(null);
    setError(null);
  };

  // Neural Network Training
  const trainNeuralNetwork = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `/api/v3/neural-network/train?session_id=${sessionId}`,
        {
          problem_type: 'binary_classification',
          framework: nnFramework,
          auto_tune: nnAutoTune,
          max_epochs: 100
        }
      );

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error training neural network');
    } finally {
      setLoading(false);
    }
  };

  // Hyperparameter Optimization
  const runOptimization = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `/api/v3/optimization/tune?session_id=${sessionId}`,
        {
          model_type: 'random_forest',
          optimizer: optimizer,
          n_trials: nTrials,
          problem_type: 'classification',
          cv_folds: 5
        }
      );

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error in optimization');
    } finally {
      setLoading(false);
    }
  };

  // Time Series Forecasting
  const runForecasting = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `/api/v3/timeseries/forecast?session_id=${sessionId}`,
        {
          method: tsMethod,
          forecast_periods: forecastPeriods,
          date_column: 'ds',
          target_column: 'y'
        }
      );

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error in forecasting');
    } finally {
      setLoading(false);
    }
  };

  // Drift Detection
  const detectDrift = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `/api/v3/mlops/detect-drift?session_id=${sessionId}`,
        {
          method: 'all',
          threshold_ks: 0.05,
          threshold_psi: 0.1
        }
      );

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error in drift detection');
    } finally {
      setLoading(false);
    }
  };

  // Model Explainability
  const explainModel = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `/api/v3/explainability/explain?session_id=${sessionId}`,
        {
          method: explainMethod,
          sample_index: parseInt(sampleIndex),
          top_k: 10
        }
      );

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error in explanation');
    } finally {
      setLoading(false);
    }
  };

  // Auto Feature Generation
  const generateFeatures = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `/api/v3/features/auto-generate?session_id=${sessionId}`
      );

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error generating features');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          🚀 Advanced ML Features
        </Typography>

        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab icon={<Psychology />} label="Neural Networks" />
          <Tab icon={<Speed />} label="Optimization" />
          <Tab icon={<Timeline />} label="Time Series" />
          <Tab icon={<BugReport />} label="Drift Detection" />
          <Tab icon={<Insights />} label="Explainability" />
          <Tab icon={<AutoAwesome />} label="Auto Features" />
        </Tabs>

        {/* Neural Networks Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Framework</InputLabel>
                <Select
                  value={nnFramework}
                  onChange={(e) => setNnFramework(e.target.value)}
                >
                  <MenuItem value="pytorch">PyTorch</MenuItem>
                  <MenuItem value="tensorflow">TensorFlow</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Auto-Tune Architecture</InputLabel>
                <Select
                  value={nnAutoTune}
                  onChange={(e) => setNnAutoTune(e.target.value)}
                >
                  <MenuItem value={false}>Manual</MenuItem>
                  <MenuItem value={true}>AutoML</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <Button
                variant="contained"
                onClick={trainNeuralNetwork}
                disabled={loading}
                fullWidth
              >
                {loading ? <CircularProgress size={24} /> : 'Train Neural Network'}
              </Button>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Optimization Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Optimizer</InputLabel>
                <Select
                  value={optimizer}
                  onChange={(e) => setOptimizer(e.target.value)}
                >
                  <MenuItem value="optuna">Optuna (TPE)</MenuItem>
                  <MenuItem value="genetic">Genetic Algorithm</MenuItem>
                  <MenuItem value="bayesian">Bayesian Optimization</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Number of Trials"
                type="number"
                value={nTrials}
                onChange={(e) => setNTrials(e.target.value)}
              />
            </Grid>

            <Grid item xs={12}>
              <Button
                variant="contained"
                onClick={runOptimization}
                disabled={loading}
                fullWidth
              >
                {loading ? <CircularProgress size={24} /> : 'Run Optimization'}
              </Button>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Time Series Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Method</InputLabel>
                <Select
                  value={tsMethod}
                  onChange={(e) => setTsMethod(e.target.value)}
                >
                  <MenuItem value="prophet">Prophet (Facebook)</MenuItem>
                  <MenuItem value="arima">ARIMA</MenuItem>
                  <MenuItem value="lstm">LSTM (Deep Learning)</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Forecast Periods"
                type="number"
                value={forecastPeriods}
                onChange={(e) => setForecastPeriods(e.target.value)}
              />
            </Grid>

            <Grid item xs={12}>
              <Button
                variant="contained"
                onClick={runForecasting}
                disabled={loading}
                fullWidth
              >
                {loading ? <CircularProgress size={24} /> : 'Generate Forecast'}
              </Button>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Drift Detection Tab */}
        <TabPanel value={tabValue} index={3}>
          <Typography variant="body2" gutterBottom>
            Detects data drift (distribution changes) and concept drift (performance degradation).
          </Typography>

          <Button
            variant="contained"
            onClick={detectDrift}
            disabled={loading}
            fullWidth
            sx={{ mt: 2 }}
          >
            {loading ? <CircularProgress size={24} /> : 'Run Drift Detection'}
          </Button>
        </TabPanel>

        {/* Explainability Tab */}
        <TabPanel value={tabValue} index={4}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Method</InputLabel>
                <Select
                  value={explainMethod}
                  onChange={(e) => setExplainMethod(e.target.value)}
                >
                  <MenuItem value="shap">SHAP Values</MenuItem>
                  <MenuItem value="lime">LIME</MenuItem>
                  <MenuItem value="whatif">What-If Analysis</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Sample Index"
                type="number"
                value={sampleIndex}
                onChange={(e) => setSampleIndex(e.target.value)}
              />
            </Grid>

            <Grid item xs={12}>
              <Button
                variant="contained"
                onClick={explainModel}
                disabled={loading}
                fullWidth
              >
                {loading ? <CircularProgress size={24} /> : 'Explain Prediction'}
              </Button>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Auto Features Tab */}
        <TabPanel value={tabValue} index={5}>
          <Typography variant="body2" gutterBottom>
            Automatically generates new features: polynomial, interactions, datetime, aggregations.
          </Typography>

          <Button
            variant="contained"
            onClick={generateFeatures}
            disabled={loading}
            fullWidth
            sx={{ mt: 2 }}
          >
            {loading ? <CircularProgress size={24} /> : 'Generate Features'}
          </Button>
        </TabPanel>

        {/* Error Display */}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        {/* Results Display */}
        {result && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Results:
            </Typography>

            <Card variant="outlined">
              <CardContent>
                <pre style={{ overflowX: 'auto' }}>
                  {JSON.stringify(result, null, 2)}
                </pre>
              </CardContent>
            </Card>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}