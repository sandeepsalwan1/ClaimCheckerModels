import path from 'path';
import { fileURLToPath } from 'url';
import CopyPlugin from 'copy-webpack-plugin';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default {
  entry: {
    background: './background.js',
    content: './content.js',
    popup: './popup.js',
    ai_models: './ai_models.js',
    test_svm: './test-svm.js'
  },
  output: {
    filename: '[name].bundle.js',
    path: path.resolve(__dirname, 'dist'),
    clean: true
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env']
          }
        }
      }
    ]
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: 'manifest.json', to: 'manifest.json' },
        { from: 'popup.html', to: 'popup.html' },
        { from: 'popup.css', to: 'popup.css' },
        { from: 'icons', to: 'icons' },
        { from: 'data', to: 'data' },
        { 
          from: 'fake-news-detection-testSet/dataset/mediaeval-2015-trainingset.csv', 
          to: 'data/mediaeval-2015-trainingset.csv' 
        },
        { 
          from: 'fake-news-detection-testSet/dataset/mediaeval-2015-testset.csv', 
          to: 'data/mediaeval-2015-testset.csv' 
        }
      ]
    })
  ]
}; 