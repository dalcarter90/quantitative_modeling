#!/usr/bin/env python3
"""
Smart ML Learning System - Market Hours Aware
==============================================
Automatically switches between stocks and crypto based on market hours:
- Market Hours (9:30 AM - 4:00 PM ET): Learn from stocks
- After Hours & Weekends: Learn from crypto
- Seamless transition with continuous learning
"""

import os
import sys
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import json
import threading
import glob
from typing import Dict, List, Tuple

class SmartMLLearner:
    def train_on_historic_data(self, start_date: str, end_date: str, mode: str = "stocks", interval: str = None):
        """Train on historic data for the given date range and mode ('stocks' or 'crypto')."""
        self.logger.info(f"[HISTORIC] Training on historic data from {start_date} to {end_date} for {mode}")
        symbols = self.get_symbols_for_mode(mode)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        current_dt = start_dt
        # Determine interval if not provided
        if interval is None:
            if (end_dt - start_dt).days <= 60:
                interval = '15m' if mode == 'crypto' else '5m'
            else:
                interval = '1d'
        while current_dt <= end_dt:
            # Fetch data for this day
            self.logger.info(f"[HISTORIC] Processing {current_dt.strftime('%Y-%m-%d')} (interval={interval})")
            data_collected = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=current_dt.strftime('%Y-%m-%d'), end=(current_dt + timedelta(days=1)).strftime('%Y-%m-%d'), interval=interval)
                    if not hist.empty:
                        # Use the same feature extraction as collect_data
                        hist['Returns'] = hist['Close'].pct_change()
                        hist['SMA_10'] = hist['Close'].rolling(window=10).mean()
                        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                        hist['Volume_MA'] = hist['Volume'].rolling(window=20).mean()
                        delta = hist['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        hist['RSI'] = 100 - (100 / (1 + rs))
                        rsi_min = hist['RSI'].rolling(window=14).min()
                        rsi_max = hist['RSI'].rolling(window=14).max()
                        hist['StochRSI'] = ((hist['RSI'] - rsi_min) / (rsi_max - rsi_min)) * 100
                        hist['StochRSI_K'] = hist['StochRSI'].rolling(window=3).mean()
                        hist['StochRSI_D'] = hist['StochRSI_K'].rolling(window=3).mean()
                        hist['BB_Middle'] = hist['Close'].rolling(window=14).mean()
                        bb_std = hist['Close'].rolling(window=14).std()
                        hist['BB_Upper'] = hist['BB_Middle'] + (bb_std * 2)
                        hist['BB_Lower'] = hist['BB_Middle'] - (bb_std * 2)
                        hist['BB_Width'] = ((hist['BB_Upper'] - hist['BB_Lower']) / hist['BB_Middle']) * 100
                        # Debug: print types and shapes
                        print("BB_Lower type:", type(hist['BB_Lower']), "shape:", getattr(hist['BB_Lower'], 'shape', None))
                        print("BB_Upper type:", type(hist['BB_Upper']), "shape:", getattr(hist['BB_Upper'], 'shape', None))
                        bb_lower = hist['BB_Lower']
                        bb_upper = hist['BB_Upper']
                        # If either is a DataFrame, select the first column
                        if isinstance(bb_lower, pd.DataFrame):
                            print("[WARNING] BB_Lower is a DataFrame, using first column only!")
                            bb_lower = bb_lower.iloc[:, 0]
                        if isinstance(bb_upper, pd.DataFrame):
                            print("[WARNING] BB_Upper is a DataFrame, using first column only!")
                            bb_upper = bb_upper.iloc[:, 0]
                        hist['BB_Position'] = ((hist['Close'] - bb_lower) / (bb_upper - bb_lower)) * 100
                        hist['Volatility'] = hist['Returns'].rolling(window=14).std()
                        # Use last row for features
                        last = hist.iloc[-1]
                        data_collected[symbol] = {
                            'price': float(last['Close']),
                            'volume': int(last['Volume']),
                            'rsi': float(last['RSI']) if not pd.isna(last['RSI']) else 50.0,
                            'stoch_rsi': float(last['StochRSI']) if not pd.isna(last['StochRSI']) else 50.0,
                            'stoch_rsi_k': float(last['StochRSI_K']) if not pd.isna(last['StochRSI_K']) else 50.0,
                            'stoch_rsi_d': float(last['StochRSI_D']) if not pd.isna(last['StochRSI_D']) else 50.0,
                            'bb_upper': float(last['BB_Upper']) if not pd.isna(last['BB_Upper']) else float(last['Close']) * 1.02,
                            'bb_lower': float(last['BB_Lower']) if not pd.isna(last['BB_Lower']) else float(last['Close']) * 0.98,
                            'bb_middle': float(last['BB_Middle']) if not pd.isna(last['BB_Middle']) else float(last['Close']),
                            'bb_width': float(last['BB_Width']) if not pd.isna(last['BB_Width']) else 4.0,
                            'bb_position': float(last['BB_Position']) if not pd.isna(last['BB_Position']) else 50.0,
                            'returns': float(last['Returns']) if not pd.isna(last['Returns']) else 0.0,
                            'volatility': float(last['Volatility']) if not pd.isna(last['Volatility']) else 0.0,
                            'sma_10': float(last['SMA_10']) if not pd.isna(last['SMA_10']) else float(last['Close']),
                            'sma_20': float(last['SMA_20']) if not pd.isna(last['SMA_20']) else float(last['Close']),
                            'timestamp': current_dt.isoformat(),
                            'data_points': len(hist),
                            'mode': mode
                        }
                except Exception as e:
                    self.logger.error(f"[HISTORIC][ERROR] {symbol}: {str(e)}")
            if data_collected:
                patterns = self.analyze_patterns(data_collected, mode)
                self.learn_from_data(data_collected, patterns, mode)
                self.logger.info(f"[HISTORIC] Learned from {len(data_collected)} symbols on {current_dt.strftime('%Y-%m-%d')}")
            else:
                self.logger.warning(f"[HISTORIC] No data collected for {current_dt.strftime('%Y-%m-%d')}")
            self.save_learning_data()
            current_dt += timedelta(days=1)
        self.logger.info(f"[HISTORIC] Finished training on historic data.")
    """Market-hours aware ML learning system with persistent learning"""
    
    def __init__(self, persistence_file: str = "smart_ml_persistent_data.json"):
        """Initialize the smart learning system with persistence"""
        
        # Stock symbols for market hours
        self.stock_symbols = [
            'OPEN','SPY', 'QQQ', 'AMD', 'NVDA', 'SNOW', 
            'RDDT', 'ANF','SMCI', 'RKLB', 'ASTS', 'MSTR', 
            'CRM', 'SHOP', 'PLTR', 'ADCT', 'NVMI', 'OKLO', 'SMR', 'GME', 'MU'
        ]
        
        # Crypto symbols for after hours
        self.crypto_symbols = [
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD',
            'XRP-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD',
            'LTC-USD', 'BCH-USD', 'ATOM-USD'
        ]
        
        # Market timezone
        self.market_tz = pytz.timezone('US/Eastern')
        
        # Persistence configuration
        self.persistence_file = persistence_file
        self.max_history_days = 90  # Keep 90 days of learning data
        
        # Learning data storage (will be loaded from file if exists)
        self.learning_data = {
            'stocks': [],
            'crypto': []
        }
        
        self.cycle_count = {'stocks': 0, 'crypto': 0}
        self.learning_metrics = {
            'pattern_accuracy': {'stocks': [], 'crypto': []},
            'total_runtime_hours': 0,
            'last_restart': datetime.now().isoformat(),
            'restart_count': 0,
            'session_start': datetime.now().isoformat(),
            'learning_milestones': []
        }
        
        self.is_running = False
        self.current_mode = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger('SmartMLLearner')
        
        # Load previous learning data
        self.load_learning_data()
        
    def load_learning_data(self):
        """Load previous learning data if available"""
        try:
            # First, try to load from main persistence file
            if os.path.exists(self.persistence_file):
                with open(self.persistence_file, 'r') as f:
                    saved_data = json.load(f)
                
                # Load learning data
                self.learning_data = saved_data.get('learning_data', {'stocks': [], 'crypto': []})
                self.cycle_count = saved_data.get('cycle_count', {'stocks': 0, 'crypto': 0})
                self.learning_metrics = saved_data.get('learning_metrics', {
                    'pattern_accuracy': {'stocks': [], 'crypto': []},
                    'total_runtime_hours': 0,
                    'last_restart': datetime.now().isoformat(),
                    'restart_count': 0,
                    'learning_milestones': []
                })
                # Ensure pattern_accuracy and subkeys always exist
                if 'pattern_accuracy' not in self.learning_metrics:
                    self.learning_metrics['pattern_accuracy'] = {'stocks': [], 'crypto': []}
                if 'stocks' not in self.learning_metrics['pattern_accuracy']:
                    self.learning_metrics['pattern_accuracy']['stocks'] = []
                if 'crypto' not in self.learning_metrics['pattern_accuracy']:
                    self.learning_metrics['pattern_accuracy']['crypto'] = []
                # Update restart metrics
                self.learning_metrics['restart_count'] += 1
                self.learning_metrics['last_restart'] = datetime.now().isoformat()
                self.learning_metrics['session_start'] = datetime.now().isoformat()
                # Clean old data (keep only recent entries)
                self.clean_old_data()
                self.logger.info(f"[LOAD] ✅ Previous learning data loaded successfully")
                self.logger.info(f"[HISTORY] 📊 Stock cycles: {self.cycle_count['stocks']}, Crypto cycles: {self.cycle_count['crypto']}")
                self.logger.info(f"[HISTORY] 📦 Stock data points: {len(self.learning_data['stocks'])}")
                self.logger.info(f"[HISTORY] 📦 Crypto data points: {len(self.learning_data['crypto'])}")
                self.logger.info(f"[HISTORY] 🔁 Previous restarts: {self.learning_metrics['restart_count'] - 1}")
                
                return True
                
            else:
                # Try to import from existing timestamped files
                self.import_existing_data()
                return False
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error loading learning data: {str(e)}")
            self.logger.info(f"[INIT] Starting with fresh learning data")
            return False
    
    def import_existing_data(self):
        """Import data from existing timestamped smart ML files in multiple locations"""
        try:
            # Search in multiple locations for learning files
            search_paths = [
                ".",  # Current directory (examples)
                "..",  # Parent directory (ai-trading-bot root)
                "../..",  # Grandparent directory (just in case)
                os.path.join("..", "smart_ml_*.json"),  # Specific pattern in parent
            ]
            
            all_files = []
            pattern = "smart_ml_learning_*.json"
            
            # Search in current directory first
            local_files = glob.glob(pattern)
            if local_files:
                all_files.extend([(f, "examples/") for f in local_files])
                self.logger.info(f"[SEARCH] 📁 Found {len(local_files)} files in examples/")
            
            # Search in parent directory (ai-trading-bot root)
            parent_pattern = os.path.join("..", pattern)
            parent_files = glob.glob(parent_pattern)
            if parent_files:
                all_files.extend([(f, "ai-trading-bot/") for f in parent_files])
                self.logger.info(f"[SEARCH] 📁 Found {len(parent_files)} files in ai-trading-bot/")
            
            # Search for any other ML-related JSON files in parent directory
            other_patterns = [
                "../smart_ml_*.json",
                "../ml_learning_*.json", 
                "../learning_*.json"
            ]
            
            for pattern_search in other_patterns:
                other_files = glob.glob(pattern_search)
                if other_files:
                    all_files.extend([(f, "ai-trading-bot/other/") for f in other_files])
                    self.logger.info(f"[SEARCH] 📁 Found {len(other_files)} additional ML files")
            
            if all_files:
                # Display all found files
                self.logger.info(f"[SEARCH] 🔍 Total files found: {len(all_files)}")
                for file_path, location in all_files:
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    self.logger.info(f"[FOUND] 📄 {location}{os.path.basename(file_path)} ({file_size:.1f}KB, {mod_time.strftime('%Y-%m-%d %H:%M')})")
                
                # Find the most recent file by modification time
                latest_file, latest_location = max(all_files, key=lambda x: os.path.getmtime(x[0]))
                self.logger.info(f"[IMPORT] 📁 Using most recent: {latest_location}{os.path.basename(latest_file)}")
                
                with open(latest_file, 'r') as f:
                    old_data = json.load(f)
                
                # Convert old format to new persistent format
                imported_cycles = {'stocks': 0, 'crypto': 0}
                imported_data_points = 0
                
                if 'learning_data' in old_data:
                    self.learning_data = old_data['learning_data']
                    imported_data_points = len(self.learning_data.get('stocks', [])) + len(self.learning_data.get('crypto', []))
                
                if 'metadata' in old_data and 'cycles' in old_data['metadata']:
                    self.cycle_count = old_data['metadata']['cycles']
                    imported_cycles = self.cycle_count
                
                self.learning_metrics['restart_count'] = 1
                self.learning_metrics['last_restart'] = datetime.now().isoformat()
                
                self.logger.info(f"[IMPORT] ✅ Successfully imported from {latest_location}")
                self.logger.info(f"[IMPORT] 📊 Stock cycles: {imported_cycles.get('stocks', 0)}, Crypto cycles: {imported_cycles.get('crypto', 0)}")
                self.logger.info(f"[IMPORT] 📦 Data points: {imported_data_points}")
                
                # Save in new persistent format
                self.save_learning_data()
                
                # Offer to clean up old files
                if len(all_files) > 1:
                    self.logger.info(f"[CLEANUP] 🧹 Note: {len(all_files)-1} other ML files found")
                    self.logger.info(f"[CLEANUP] 💡 Consider consolidating old files after verifying import")
                
            else:
                self.logger.info(f"[INIT] 🆕 No previous learning data found in any location, starting fresh")
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error importing existing data: {str(e)}")
    
    def clean_old_data(self):
        """Remove data older than max_history_days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
            
            for mode in ['stocks', 'crypto']:
                original_count = len(self.learning_data[mode])
                self.learning_data[mode] = [
                    entry for entry in self.learning_data[mode] 
                    if datetime.fromisoformat(entry['timestamp']) > cutoff_date
                ]
                cleaned_count = len(self.learning_data[mode])
                
                if original_count > cleaned_count:
                    self.logger.info(f"[CLEANUP] 🧹 {mode}: Removed {original_count - cleaned_count} old entries")
        except Exception as e:
            self.logger.error(f"[ERROR] Error cleaning old data: {str(e)}")
    
    def save_learning_data(self, backup: bool = False):
        """Save learning data with optional backup"""
        try:
            # Calculate session runtime
            session_start = datetime.fromisoformat(self.learning_metrics['session_start'])
            session_hours = (datetime.now() - session_start).total_seconds() / 3600
            self.learning_metrics['total_runtime_hours'] += session_hours
            
            save_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'cycles': self.cycle_count,
                    'modes': ['stocks', 'crypto'],
                    'stock_symbols': self.stock_symbols,
                    'crypto_symbols': self.crypto_symbols,
                    'version': '2.0_persistent'
                },
                'cycle_count': self.cycle_count,
                'learning_data': self.learning_data,
                'learning_metrics': self.learning_metrics
            }
            
            # Save main persistence file
            with open(self.persistence_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            # Optional backup with timestamp
            if backup:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_filename = f"smart_ml_backup_{timestamp}.json"
                with open(backup_filename, 'w') as f:
                    json.dump(save_data, f, indent=2)
                self.logger.info(f"[BACKUP] 💾 Learning data backed up to {backup_filename}")
            
            self.logger.info(f"[SAVE] 💾 Learning data saved to {self.persistence_file}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error saving learning data: {str(e)}")
    
    def get_learning_statistics(self) -> Dict:
        """Get comprehensive learning statistics"""
        total_cycles = sum(self.cycle_count.values())
        total_data_points = len(self.learning_data['stocks']) + len(self.learning_data['crypto'])
        
        # Calculate learning maturity and readiness
        learning_maturity = min(total_cycles / 500, 1.0)  # 0-1 scale, mature at 500 cycles
        
        # Calculate recent accuracy if available
        recent_stock_accuracy = 0.5
        recent_crypto_accuracy = 0.5
        
        if len(self.learning_metrics['pattern_accuracy']['stocks']) > 0:
            recent_stock_accuracy = np.mean(self.learning_metrics['pattern_accuracy']['stocks'][-10:])
        if len(self.learning_metrics['pattern_accuracy']['crypto']) > 0:
            recent_crypto_accuracy = np.mean(self.learning_metrics['pattern_accuracy']['crypto'][-10:])
        
        # Readiness assessment
        is_ready_for_paper_trading = (
            total_cycles >= 200 and 
            learning_maturity >= 0.4 and
            total_data_points >= 100
        )
        
        return {
            'total_cycles': total_cycles,
            'stock_cycles': self.cycle_count['stocks'],
            'crypto_cycles': self.cycle_count['crypto'],
            'total_data_points': total_data_points,
            'stock_data_points': len(self.learning_data['stocks']),
            'crypto_data_points': len(self.learning_data['crypto']),
            'restart_count': self.learning_metrics['restart_count'],
            'total_runtime_hours': self.learning_metrics['total_runtime_hours'],
            'recent_stock_accuracy': recent_stock_accuracy,
            'recent_crypto_accuracy': recent_crypto_accuracy,
            'learning_maturity': learning_maturity,
            'is_ready_for_paper_trading': is_ready_for_paper_trading,
            'readiness_percentage': min(learning_maturity * 100, 100)
        }
    
    def consolidate_all_learning_data(self):
        """Consolidate all learning files from multiple locations into persistent format"""
        self.logger.info("[CONSOLIDATE] 🔄 Starting comprehensive data consolidation")
        
        # Search all possible locations and patterns
        search_locations = [
            (".", "smart_ml_learning_*.json", "examples/"),
            ("..", "smart_ml_learning_*.json", "ai-trading-bot/"),
            (".", "smart_ml_*.json", "examples/other"),
            ("..", "smart_ml_*.json", "ai-trading-bot/other"),
            ("..", "ml_learning_*.json", "ai-trading-bot/ml"),
        ]
        
        all_learning_files = []
        
        for location, pattern, description in search_locations:
            search_pattern = os.path.join(location, pattern) if location != "." else pattern
            found_files = glob.glob(search_pattern)
            
            if found_files:
                for file_path in found_files:
                    file_info = {
                        'path': file_path,
                        'location': description,
                        'size_kb': os.path.getsize(file_path) / 1024,
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                        'basename': os.path.basename(file_path)
                    }
                    all_learning_files.append(file_info)
                
                self.logger.info(f"[SEARCH] 📁 {description}: Found {len(found_files)} files")
        
        if not all_learning_files:
            self.logger.info("[CONSOLIDATE] ❌ No learning files found to consolidate")
            return False
        
        # Sort by modification time (newest first)
        all_learning_files.sort(key=lambda x: x['modified'], reverse=True)
        
        self.logger.info(f"[CONSOLIDATE] 📊 Found {len(all_learning_files)} total learning files:")
        for i, file_info in enumerate(all_learning_files[:10]):  # Show top 10
            age_hours = (datetime.now() - file_info['modified']).total_seconds() / 3600
            self.logger.info(f"   {i+1}. {file_info['location']}{file_info['basename']} ({file_info['size_kb']:.1f}KB, {age_hours:.1f}h ago)")
        
        if len(all_learning_files) > 10:
            self.logger.info(f"   ... and {len(all_learning_files) - 10} more files")
        
        # Consolidate data from all files
        consolidated_data = {'stocks': [], 'crypto': []}
        consolidated_cycles = {'stocks': 0, 'crypto': 0}
        files_processed = 0
        
        for file_info in all_learning_files:
            try:
                with open(file_info['path'], 'r') as f:
                    data = json.load(f)
                
                # Extract learning data
                if 'learning_data' in data:
                    for mode in ['stocks', 'crypto']:
                        if mode in data['learning_data']:
                            new_entries = data['learning_data'][mode]
                            consolidated_data[mode].extend(new_entries)
                
                # Extract cycle counts (use maximum found)
                if 'metadata' in data and 'cycles' in data['metadata']:
                    cycles = data['metadata']['cycles']
                    for mode in ['stocks', 'crypto']:
                        if mode in cycles:
                            consolidated_cycles[mode] = max(consolidated_cycles[mode], cycles[mode])
                
                files_processed += 1
                
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to process {file_info['basename']}: {str(e)}")
        
        # Remove duplicates based on timestamp and symbol
        for mode in ['stocks', 'crypto']:
            unique_entries = []
            seen_entries = set()
            
            for entry in consolidated_data[mode]:
                # Create unique key from timestamp and symbols
                if 'data' in entry:
                    symbols = sorted(entry['data'].keys())
                    key = f"{entry['timestamp']}_{hash(tuple(symbols))}"
                    
                    if key not in seen_entries:
                        seen_entries.add(key)
                        unique_entries.append(entry)
            
            consolidated_data[mode] = unique_entries
        
        # Update internal data
        self.learning_data = consolidated_data
        self.cycle_count = consolidated_cycles
        self.learning_metrics['restart_count'] = len(all_learning_files)  # Rough estimate
        
        total_data_points = len(consolidated_data['stocks']) + len(consolidated_data['crypto'])
        
        self.logger.info(f"[CONSOLIDATE] ✅ Consolidation complete!")
        self.logger.info(f"[CONSOLIDATE] 📊 Stock cycles: {consolidated_cycles['stocks']}")
        self.logger.info(f"[CONSOLIDATE] 🪙 Crypto cycles: {consolidated_cycles['crypto']}")
        self.logger.info(f"[CONSOLIDATE] 📦 Total data points: {total_data_points}")
        self.logger.info(f"[CONSOLIDATE] 📄 Files processed: {files_processed}")
        
        # Save consolidated data
        self.save_learning_data()
        
        # Return consolidated information
        return {
            'total_stock_cycles': consolidated_cycles['stocks'],
            'total_crypto_cycles': consolidated_cycles['crypto'],
            'total_cycles': consolidated_cycles['stocks'] + consolidated_cycles['crypto'],
            'learning_data': self.learning_data,
            'files_processed': files_processed,
            'sources': {
                'examples/': len([f for f in all_learning_files if 'examples/' in f['location']]),
                'ai-trading-bot/': len([f for f in all_learning_files if 'ai-trading-bot/' in f['location']])
            }
        }
        
    def is_market_hours(self) -> bool:
        """Check if currently in stock market hours"""
        now = datetime.now(self.market_tz)
        
        # Market is closed on weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def get_next_market_event(self) -> Tuple[str, datetime]:
        """Get the next market open/close event"""
        now = datetime.now(self.market_tz)
        
        if self.is_market_hours():
            # Market is open, next event is close
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            if now > market_close:
                # Already past close, next is tomorrow's open
                next_day = now + timedelta(days=1)
                while next_day.weekday() >= 5:  # Skip weekends
                    next_day += timedelta(days=1)
                market_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
                return "market_open", market_open
            else:
                return "market_close", market_close
        else:
            # Market is closed, next event is open
            if now.weekday() >= 5:  # Weekend
                # Find next Monday
                days_until_monday = 7 - now.weekday()
                next_monday = now + timedelta(days=days_until_monday)
                market_open = next_monday.replace(hour=9, minute=30, second=0, microsecond=0)
            else:
                # Weekday but after hours
                if now.hour >= 16:
                    # After close, next is tomorrow
                    next_day = now + timedelta(days=1)
                    market_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
                else:
                    # Before open today
                    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            
            return "market_open", market_open
    
    def determine_learning_mode(self) -> str:
        """Determine whether to learn stocks or crypto"""
        if self.is_market_hours():
            return "stocks"
        else:
            return "crypto"
    
    def get_symbols_for_mode(self, mode: str) -> List[str]:
        """Get symbols for the current learning mode"""
        if mode == "stocks":
            return self.stock_symbols
        else:
            return self.crypto_symbols
    
    def collect_data(self, symbols: List[str], mode: str) -> Dict:
        """Collect market data for given symbols"""
        self.logger.info(f"[{mode.upper()}] Collecting data for {len(symbols)} symbols...")
        
        data_collected = {}
        
        # Determine appropriate period and interval based on mode
        if mode == "stocks":
            period = "1d"
            interval = "5m"
        else:  # crypto
            period = "1d"
            interval = "15m"  # Crypto is 24/7, can use longer intervals
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                
                if not hist.empty:
                    # Calculate indicators
                    hist['Returns'] = hist['Close'].pct_change()
                    hist['SMA_10'] = hist['Close'].rolling(window=10).mean()
                    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                    hist['Volume_MA'] = hist['Volume'].rolling(window=20).mean()
                    
                    # RSI calculation
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    hist['RSI'] = 100 - (100 / (1 + rs))
                    
                    # StochasticRSI calculation
                    rsi_min = hist['RSI'].rolling(window=14).min()
                    rsi_max = hist['RSI'].rolling(window=14).max()
                    hist['StochRSI'] = ((hist['RSI'] - rsi_min) / (rsi_max - rsi_min)) * 100
                    hist['StochRSI_K'] = hist['StochRSI'].rolling(window=3).mean()  # %K smoothing
                    hist['StochRSI_D'] = hist['StochRSI_K'].rolling(window=3).mean()  # %D smoothing
                    
                    # Bollinger Bands calculation
                    hist['BB_Middle'] = hist['Close'].rolling(window=14).mean()
                    bb_std = hist['Close'].rolling(window=14).std()
                    hist['BB_Upper'] = hist['BB_Middle'] + (bb_std * 2)
                    hist['BB_Lower'] = hist['BB_Middle'] - (bb_std * 2)
                    hist['BB_Width'] = ((hist['BB_Upper'] - hist['BB_Lower']) / hist['BB_Middle']) * 100
                    hist['BB_Position'] = ((hist['Close'] - hist['BB_Lower']) / (hist['BB_Upper'] - hist['BB_Lower'])) * 100
                    
                    # Volatility
                    hist['Volatility'] = hist['Returns'].rolling(window=14).std()
                    
                    # Extract current values
                    current_price = float(hist['Close'].iloc[-1])
                    current_volume = int(hist['Volume'].iloc[-1])
                    current_rsi = float(hist['RSI'].iloc[-1]) if not pd.isna(hist['RSI'].iloc[-1]) else 50.0
                    current_returns = float(hist['Returns'].iloc[-1]) if not pd.isna(hist['Returns'].iloc[-1]) else 0.0
                    current_volatility = float(hist['Volatility'].iloc[-1]) if not pd.isna(hist['Volatility'].iloc[-1]) else 0.0
                    sma_10 = float(hist['SMA_10'].iloc[-1]) if not pd.isna(hist['SMA_10'].iloc[-1]) else current_price
                    sma_20 = float(hist['SMA_20'].iloc[-1]) if not pd.isna(hist['SMA_20'].iloc[-1]) else current_price
                    
                    # Extract StochasticRSI values
                    stoch_rsi = float(hist['StochRSI'].iloc[-1]) if not pd.isna(hist['StochRSI'].iloc[-1]) else 50.0
                    stoch_rsi_k = float(hist['StochRSI_K'].iloc[-1]) if not pd.isna(hist['StochRSI_K'].iloc[-1]) else 50.0
                    stoch_rsi_d = float(hist['StochRSI_D'].iloc[-1]) if not pd.isna(hist['StochRSI_D'].iloc[-1]) else 50.0
                    
                    # Extract Bollinger Bands values
                    bb_upper = float(hist['BB_Upper'].iloc[-1]) if not pd.isna(hist['BB_Upper'].iloc[-1]) else current_price * 1.02
                    bb_lower = float(hist['BB_Lower'].iloc[-1]) if not pd.isna(hist['BB_Lower'].iloc[-1]) else current_price * 0.98
                    bb_middle = float(hist['BB_Middle'].iloc[-1]) if not pd.isna(hist['BB_Middle'].iloc[-1]) else current_price
                    bb_width = float(hist['BB_Width'].iloc[-1]) if not pd.isna(hist['BB_Width'].iloc[-1]) else 4.0
                    bb_position = float(hist['BB_Position'].iloc[-1]) if not pd.isna(hist['BB_Position'].iloc[-1]) else 50.0
                    
                    data_collected[symbol] = {
                        'price': current_price,
                        'volume': current_volume,
                        'rsi': current_rsi,
                        'stoch_rsi': stoch_rsi,
                        'stoch_rsi_k': stoch_rsi_k,
                        'stoch_rsi_d': stoch_rsi_d,
                        'bb_upper': bb_upper,
                        'bb_lower': bb_lower,
                        'bb_middle': bb_middle,
                        'bb_width': bb_width,
                        'bb_position': bb_position,
                        'returns': current_returns,
                        'volatility': current_volatility,
                        'sma_10': sma_10,
                        'sma_20': sma_20,
                        'timestamp': datetime.now().isoformat(),
                        'data_points': len(hist),
                        'mode': mode
                    }
                    
                    # Display with appropriate currency symbol and new indicators
                    if mode == "crypto":
                        self.logger.info(f"[SUCCESS] {symbol}: ${current_price:.4f} (RSI: {current_rsi:.1f}, StochRSI: {stoch_rsi:.1f}, BB: {bb_position:.1f}%) Vol: {current_volatility:.3f}")
                    else:
                        self.logger.info(f"[SUCCESS] {symbol}: ${current_price:.2f} (RSI: {current_rsi:.1f}, StochRSI: {stoch_rsi:.1f}, BB: {bb_position:.1f}%)")
                    
            except Exception as e:
                self.logger.error(f"[ERROR] Error with {symbol}: {str(e)}")
        
        return data_collected
    
    def analyze_patterns(self, data: Dict, mode: str) -> Dict:
        """Analyze patterns specific to the asset class"""
        self.logger.info(f"[{mode.upper()}] Analyzing market patterns...")
        
        patterns = {}
        
        for symbol, info in data.items():
            try:
                price = info['price']
                rsi = info['rsi']
                sma_10 = info['sma_10']
                sma_20 = info['sma_20']
                volatility = info['volatility']
                
                # Extract new indicators
                stoch_rsi = info['stoch_rsi']
                stoch_rsi_k = info['stoch_rsi_k']
                stoch_rsi_d = info['stoch_rsi_d']
                bb_position = info['bb_position']
                bb_width = info['bb_width']
                
                # Determine trend
                if price > sma_20 and sma_10 > sma_20:
                    trend = "BULLISH"
                elif price < sma_20 and sma_10 < sma_20:
                    trend = "BEARISH"
                else:
                    trend = "NEUTRAL"
                
                # RSI analysis (adjusted thresholds for crypto)
                if mode == "crypto":
                    # Crypto is more volatile, adjust thresholds
                    if rsi > 75:
                        rsi_signal = "OVERBOUGHT"
                    elif rsi < 25:
                        rsi_signal = "OVERSOLD"
                    else:
                        rsi_signal = "NORMAL"
                else:
                    # Traditional stock thresholds
                    if rsi > 70:
                        rsi_signal = "OVERBOUGHT"
                    elif rsi < 30:
                        rsi_signal = "OVERSOLD"
                    else:
                        rsi_signal = "NORMAL"
                
                # StochasticRSI analysis
                if mode == "crypto":
                    # More aggressive thresholds for crypto
                    if stoch_rsi > 80:
                        stoch_signal = "OVERBOUGHT"
                    elif stoch_rsi < 20:
                        stoch_signal = "OVERSOLD"
                    else:
                        stoch_signal = "NORMAL"
                else:
                    # Traditional thresholds for stocks
                    if stoch_rsi > 80:
                        stoch_signal = "OVERBOUGHT"
                    elif stoch_rsi < 20:
                        stoch_signal = "OVERSOLD"
                    else:
                        stoch_signal = "NORMAL"
                
                # Bollinger Bands analysis
                if bb_position > 80:
                    bb_signal = "UPPER_BAND"
                elif bb_position < 20:
                    bb_signal = "LOWER_BAND"
                elif bb_position > 50:
                    bb_signal = "ABOVE_MIDDLE"
                else:
                    bb_signal = "BELOW_MIDDLE"
                
                # Bollinger Bands squeeze detection
                bb_squeeze = "SQUEEZE" if bb_width < 10 else "NORMAL"
                
                # Volatility analysis
                if mode == "crypto":
                    vol_threshold = 0.05  # 5% for crypto
                else:
                    vol_threshold = 0.02  # 2% for stocks
                
                vol_signal = "HIGH" if volatility > vol_threshold else "NORMAL"
                
                patterns[symbol] = {
                    'trend': trend,
                    'rsi_signal': rsi_signal,
                    'stoch_rsi_signal': stoch_signal,
                    'bb_signal': bb_signal,
                    'bb_squeeze': bb_squeeze,
                    'volatility_signal': vol_signal,
                    'momentum': 'STRONG' if abs(info['returns']) > vol_threshold else 'WEAK',
                    'confidence': min(abs(rsi - 50) / 25, 1.0),
                    'stoch_momentum': 'BULLISH' if stoch_rsi_k > stoch_rsi_d else 'BEARISH'
                }
                
                self.logger.info(f"[PATTERN] {symbol}: {trend} trend, RSI {rsi_signal}, StochRSI {stoch_signal}, BB {bb_signal}, Vol {vol_signal}")
                
            except Exception as e:
                self.logger.error(f"[ERROR] Pattern analysis error for {symbol}: {str(e)}")
        
        return patterns
    
    def get_trading_prediction(self, symbol: str) -> Dict:
        """Get trading prediction for a specific symbol based on ML learning"""
        try:
            # Get current market data
            is_market_hours = self.is_market_hours()
            
            # Fetch recent data for the symbol
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            # Get 5 days of data with 1-minute intervals for recent analysis
            hist_data = ticker.history(period="5d", interval="5m")
            
            if hist_data.empty:
                return {'action': 'HOLD', 'confidence': 0.0, 'reasoning': 'No data available'}
            
            # Calculate technical indicators
            tech_indicators = self.calculate_technical_indicators(symbol, hist_data)
            
            # Get the most recent data point
            latest = tech_indicators.iloc[-1]
            
            # Transform to expected format for analyze_patterns
            data_point = {
                symbol: {
                    'price': float(latest['Close']),
                    'rsi': float(latest['RSI']) if not pd.isna(latest['RSI']) else 50.0,
                    'sma_10': float(latest['SMA_10']) if not pd.isna(latest['SMA_10']) else float(latest['Close']),
                    'sma_20': float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else float(latest['Close']),
                    'sma_50': float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else float(latest['Close']),
                    'volatility': float(latest['Volatility']) if not pd.isna(latest['Volatility']) else 0.02,
                    'volume': float(latest['Volume']),
                    'returns': float((latest['Close'] - tech_indicators.iloc[-2]['Close']) / tech_indicators.iloc[-2]['Close']) if len(tech_indicators) > 1 else 0.0,
                    'stoch_rsi': float(latest['StochRSI']) if not pd.isna(latest['StochRSI']) else 50.0,
                    'stoch_rsi_k': float(latest['StochRSI']) if not pd.isna(latest['StochRSI']) else 50.0,
                    'stoch_rsi_d': float(latest['StochRSI']) if not pd.isna(latest['StochRSI']) else 50.0,
                    'bb_position': float((latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])) if not pd.isna(latest['BB_Upper']) else 0.5,
                    'bb_width': float(latest['BB_Width']) if not pd.isna(latest['BB_Width']) else 10.0,
                    'macd': float(latest['MACD']) if not pd.isna(latest['MACD']) else 0.0,
                    'macd_signal': float(latest['MACD_Signal']) if not pd.isna(latest['MACD_Signal']) else 0.0
                }
            }
            
            # Determine mode based on symbol and market hours
            if symbol.endswith('-USD'):
                mode = "crypto"
            else:
                mode = "stocks" if is_market_hours else "crypto"
            
            # Use analyze_patterns to get current market state
            patterns = self.analyze_patterns(data_point, mode)
            
            if symbol not in patterns:
                return {'action': 'HOLD', 'confidence': 0.0, 'reasoning': 'Pattern analysis failed'}
            
            pattern = patterns[symbol]
            
            # ML-based prediction logic using learned patterns
            prediction = self.generate_ml_prediction(symbol, pattern, tech_indicators, mode)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ Error getting prediction for {symbol}: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'reasoning': f'Error: {e}'}
    
    def generate_ml_prediction(self, symbol: str, pattern: Dict, tech_data: Dict, mode: str) -> Dict:
        """Generate ML-based trading prediction using learned patterns"""
        
        # Base confidence from pattern recognition
        base_confidence = pattern.get('confidence', 0.5)
        
        # Initialize prediction components
        buy_signals = 0
        sell_signals = 0
        signal_strength = 0
        reasoning_parts = []
        
        # Trend Analysis (40% weight)
        if pattern['trend'] == 'BULLISH':
            buy_signals += 0.4
            reasoning_parts.append("Bullish trend")
        elif pattern['trend'] == 'BEARISH':
            sell_signals += 0.4
            reasoning_parts.append("Bearish trend")
        
        # RSI Analysis (20% weight)
        if pattern['rsi_signal'] == 'OVERSOLD':
            buy_signals += 0.2
            reasoning_parts.append("RSI oversold")
        elif pattern['rsi_signal'] == 'OVERBOUGHT':
            sell_signals += 0.2
            reasoning_parts.append("RSI overbought")
        
        # StochasticRSI Analysis (20% weight)
        if pattern['stoch_rsi_signal'] == 'OVERSOLD' and pattern['stoch_momentum'] == 'BULLISH':
            buy_signals += 0.2
            reasoning_parts.append("StochRSI bullish divergence")
        elif pattern['stoch_rsi_signal'] == 'OVERBOUGHT' and pattern['stoch_momentum'] == 'BEARISH':
            sell_signals += 0.2
            reasoning_parts.append("StochRSI bearish divergence")
        
        # Bollinger Bands Analysis (15% weight)
        if pattern['bb_signal'] == 'LOWER_BAND' and pattern['bb_squeeze'] == 'NORMAL':
            buy_signals += 0.15
            reasoning_parts.append("BB bounce potential")
        elif pattern['bb_signal'] == 'UPPER_BAND' and pattern['bb_squeeze'] == 'NORMAL':
            sell_signals += 0.15
            reasoning_parts.append("BB resistance")
        
        # Volatility & Momentum (5% weight)
        if pattern['momentum'] == 'STRONG' and pattern['volatility_signal'] == 'HIGH':
            if buy_signals > sell_signals:
                buy_signals += 0.05
                reasoning_parts.append("Strong momentum")
            else:
                sell_signals += 0.05
                reasoning_parts.append("Strong momentum reversal")
        
        # Apply learning-based confidence adjustment
        learning_stats = self.get_learning_statistics()
        maturity_multiplier = min(learning_stats['learning_maturity'], 1.0)
        
        # Calculate final confidence
        signal_strength = abs(buy_signals - sell_signals)
        final_confidence = min(base_confidence * signal_strength * maturity_multiplier, 0.95)
        
        # Determine action
        if buy_signals > sell_signals and final_confidence > 0.6:
            action = 'BUY'
        elif sell_signals > buy_signals and final_confidence > 0.6:
            action = 'SELL'
        else:
            action = 'HOLD'
            final_confidence = max(final_confidence, 0.3)  # Minimum confidence for HOLD
        
        # Calculate target price and stop loss (basic implementation)
        current_price = tech_data.get('price', 0)
        target_price = None
        stop_loss = None
        
        if action == 'BUY' and current_price > 0:
            target_price = current_price * 1.02  # 2% target
            stop_loss = current_price * 0.98     # 2% stop loss
        elif action == 'SELL' and current_price > 0:
            target_price = current_price * 0.98  # 2% target (for short)
            stop_loss = current_price * 1.02     # 2% stop loss (for short)
        
        # Create reasoning string
        reasoning = f"ML Analysis ({learning_stats['total_cycles']} cycles): " + ", ".join(reasoning_parts)
        
        prediction = {
            'action': action,
            'confidence': final_confidence,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'reasoning': reasoning,
            'pattern_summary': pattern,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'learning_maturity': learning_stats['learning_maturity']
        }
        
        self.logger.info(f"[PREDICTION] {symbol}: {action} (confidence: {final_confidence:.1%}) - {reasoning}")
        
        return prediction
    
    def get_portfolio_signals(self, symbols: List[str] = None) -> List[Dict]:
        """Get trading signals for multiple symbols"""
        if symbols is None:
            is_market_hours = self.is_market_hours()
            symbols = self.stock_symbols[:5] if is_market_hours else self.crypto_symbols[:5]
        
        signals = []
        
        for symbol in symbols:
            try:
                prediction = self.get_trading_prediction(symbol)
                if prediction['confidence'] > 0.7:  # High confidence threshold
                    signals.append({
                        'symbol': symbol,
                        'prediction': prediction,
                        'timestamp': datetime.now()
                    })
            except Exception as e:
                self.logger.error(f"❌ Error getting signal for {symbol}: {e}")
                continue
        
        # Sort by confidence
        signals.sort(key=lambda x: x['prediction']['confidence'], reverse=True)
        
        return signals
    
    def calculate_technical_indicators(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for a dataframe"""
        if df.empty or len(df) < 20:
            return df
        
        try:
            # Close price for calculations
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            # RSI (14-period)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # StochasticRSI
            rsi = df['RSI']
            stoch_rsi = (rsi - rsi.rolling(window=14).min()) / (rsi.rolling(window=14).max() - rsi.rolling(window=14).min())
            df['StochRSI'] = stoch_rsi * 100
            
            # Bollinger Bands (20-period, 2 std)
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Lower'] = sma_20 - (std_20 * 2)
            df['BB_Middle'] = sma_20
            df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
            
            # Moving Averages
            df['SMA_10'] = close.rolling(window=10).mean()
            df['SMA_20'] = close.rolling(window=20).mean()
            df['SMA_50'] = close.rolling(window=50).mean() if len(df) >= 50 else close.rolling(window=len(df)).mean()
            
            # MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Volatility (20-period rolling std)
            df['Volatility'] = close.pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # Volume indicators
            df['Volume_SMA'] = volume.rolling(window=20).mean()
            df['Volume_Ratio'] = volume / df['Volume_SMA']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return df
    
    def learn_from_data(self, data: Dict, patterns: Dict, mode: str):
        """Learn from data with mode-specific insights"""
        self.logger.info(f"[{mode.upper()}] Learning from market data...")
        
        # Create learning entry
        learning_entry = {
            'timestamp': datetime.now().isoformat(),
            'cycle': self.cycle_count[mode] + 1,
            'mode': mode,
            'data': data,
            'patterns': patterns,
            'market_summary': self.summarize_market(data, patterns, mode),
            'market_context': self.get_market_context()
        }
        
        self.learning_data[mode].append(learning_entry)
        
        # Keep only last 100 entries per mode
        if len(self.learning_data[mode]) > 100:
            self.learning_data[mode] = self.learning_data[mode][-100:]
        
        # Cross-mode learning insights
        self.analyze_cross_mode_patterns()
    
    def summarize_market(self, data: Dict, patterns: Dict, mode: str) -> Dict:
        """Create market summary with mode-specific metrics"""
        bullish_count = sum(1 for p in patterns.values() if p['trend'] == 'BULLISH')
        bearish_count = sum(1 for p in patterns.values() if p['trend'] == 'BEARISH')
        overbought_count = sum(1 for p in patterns.values() if p['rsi_signal'] == 'OVERBOUGHT')
        oversold_count = sum(1 for p in patterns.values() if p['rsi_signal'] == 'OVERSOLD')
        high_vol_count = sum(1 for p in patterns.values() if p['volatility_signal'] == 'HIGH')
        
        # New indicator counts
        stoch_overbought_count = sum(1 for p in patterns.values() if p['stoch_rsi_signal'] == 'OVERBOUGHT')
        stoch_oversold_count = sum(1 for p in patterns.values() if p['stoch_rsi_signal'] == 'OVERSOLD')
        bb_upper_count = sum(1 for p in patterns.values() if p['bb_signal'] == 'UPPER_BAND')
        bb_lower_count = sum(1 for p in patterns.values() if p['bb_signal'] == 'LOWER_BAND')
        bb_squeeze_count = sum(1 for p in patterns.values() if p['bb_squeeze'] == 'SQUEEZE')
        
        rsi_values = [data[s]['rsi'] for s in data.keys() if not pd.isna(data[s]['rsi'])]
        stoch_rsi_values = [data[s]['stoch_rsi'] for s in data.keys() if not pd.isna(data[s]['stoch_rsi'])]
        bb_width_values = [data[s]['bb_width'] for s in data.keys() if not pd.isna(data[s]['bb_width'])]
        vol_values = [data[s]['volatility'] for s in data.keys() if not pd.isna(data[s]['volatility'])]
        
        avg_rsi = float(np.mean(rsi_values)) if rsi_values else 50.0
        avg_stoch_rsi = float(np.mean(stoch_rsi_values)) if stoch_rsi_values else 50.0
        avg_bb_width = float(np.mean(bb_width_values)) if bb_width_values else 10.0
        avg_vol = float(np.mean(vol_values)) if vol_values else 0.0
        avg_volume = float(np.mean([data[s]['volume'] for s in data.keys()]))
        
        return {
            'mode': mode,
            'bullish_symbols': bullish_count,
            'bearish_symbols': bearish_count,
            'overbought_symbols': overbought_count,
            'oversold_symbols': oversold_count,
            'stoch_overbought_symbols': stoch_overbought_count,
            'stoch_oversold_symbols': stoch_oversold_count,
            'bb_upper_symbols': bb_upper_count,
            'bb_lower_symbols': bb_lower_count,
            'bb_squeeze_symbols': bb_squeeze_count,
            'high_volatility_symbols': high_vol_count,
            'average_rsi': avg_rsi,
            'average_stoch_rsi': avg_stoch_rsi,
            'average_bb_width': avg_bb_width,
            'average_volatility': avg_vol,
            'average_volume': avg_volume,
            'market_sentiment': 'BULLISH' if bullish_count > bearish_count else 'BEARISH',
            'volatility_regime': 'HIGH' if avg_vol > (0.05 if mode == 'crypto' else 0.02) else 'NORMAL'
        }
    
    def get_market_context(self) -> Dict:
        """Get current market context"""
        now = datetime.now(self.market_tz)
        next_event, next_time = self.get_next_market_event()
        
        return {
            'current_time': now.isoformat(),
            'is_market_hours': self.is_market_hours(),
            'weekday': now.strftime('%A'),
            'next_event': next_event,
            'next_event_time': next_time.isoformat(),
            'time_until_next_event': str(next_time - now)
        }
    
    def analyze_cross_mode_patterns(self):
        """Analyze patterns across stocks and crypto"""
        if len(self.learning_data['stocks']) < 2 or len(self.learning_data['crypto']) < 2:
            return
        
        self.logger.info("[CROSS-MODE] Analyzing correlations between stocks and crypto...")
        
        # Get recent data
        recent_stocks = self.learning_data['stocks'][-5:]
        recent_crypto = self.learning_data['crypto'][-5:]
        
        # Analyze sentiment correlation
        stock_bullish_avg = np.mean([d['market_summary']['bullish_symbols'] for d in recent_stocks])
        crypto_bullish_avg = np.mean([d['market_summary']['bullish_symbols'] for d in recent_crypto])
        
        correlation = "POSITIVE" if (stock_bullish_avg > 5) == (crypto_bullish_avg > 7) else "NEGATIVE"
        
        self.logger.info(f"[CORRELATION] Stock-Crypto sentiment: {correlation}")
        self.logger.info(f"[METRICS] Stocks bullish: {stock_bullish_avg:.1f}, Crypto bullish: {crypto_bullish_avg:.1f}")
    
    def learning_cycle(self):
        """Execute a single learning cycle"""
        # Determine current mode
        mode = self.determine_learning_mode()
        symbols = self.get_symbols_for_mode(mode)
        
        # Check if mode changed
        if self.current_mode != mode:
            if self.current_mode is not None:
                next_event, next_time = self.get_next_market_event()
                self.logger.info(f"[MODE SWITCH] Switching from {self.current_mode} to {mode}")
                self.logger.info(f"[SCHEDULE] Next switch at {next_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            self.current_mode = mode
        
        self.cycle_count[mode] += 1
        cycle_start = datetime.now()
        
        self.logger.info(f"[CYCLE] {mode.title()} Learning Cycle #{self.cycle_count[mode]}")
        self.logger.info("-" * 50)
        
        try:
            # Collect data
            data = self.collect_data(symbols, mode)
            
            if data:
                # Analyze patterns
                patterns = self.analyze_patterns(data, mode)
                
                # Learn from data
                self.learn_from_data(data, patterns, mode)
                
                # Save progress every 10 cycles (combined)
                total_cycles = sum(self.cycle_count.values())
                if total_cycles % 10 == 0:
                    self.save_learning_data()
                
                cycle_time = (datetime.now() - cycle_start).total_seconds()
                self.logger.info(f"[SUCCESS] {mode.title()} cycle completed in {cycle_time:.1f} seconds")
                self.logger.info(f"[STATS] Total cycles - Stocks: {self.cycle_count['stocks']}, Crypto: {self.cycle_count['crypto']}")
                
                return True
            else:
                self.logger.warning(f"[WARNING] No {mode} data collected, skipping cycle")
                return False
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error in {mode} learning cycle: {str(e)}")
            return False
    
    def start_smart_learning(self, base_interval_minutes: int = 15):
        """Start smart learning with mode switching and persistence"""
        
        # Get learning statistics
        stats = self.get_learning_statistics()
        
        self.logger.info("[START] Persistent Smart ML Learning System")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Stock symbols: {len(self.stock_symbols)} stocks")
        self.logger.info(f"🪙 Crypto symbols: {len(self.crypto_symbols)} cryptocurrencies")
        self.logger.info(f"⏰ Base interval: {base_interval_minutes} minutes")
        self.logger.info(f"💾 Persistence: Enabled ({self.persistence_file})")
        self.logger.info("")
        self.logger.info("📈 LEARNING STATISTICS:")
        self.logger.info(f"   🔄 Total cycles: {stats['total_cycles']}")
        self.logger.info(f"   📊 Stock cycles: {stats['stock_cycles']}")
        self.logger.info(f"   🪙 Crypto cycles: {stats['crypto_cycles']}")
        self.logger.info(f"   📦 Data points: {stats['total_data_points']}")
        self.logger.info(f"   🔁 Previous restarts: {stats['restart_count'] - 1}")
        self.logger.info(f"   ⏱️  Total runtime: {stats['total_runtime_hours']:.1f} hours")
        self.logger.info(f"   🎯 Learning maturity: {stats['learning_maturity']:.1%}")
        self.logger.info(f"   📈 Readiness: {stats['readiness_percentage']:.0f}%")
        
        if stats['total_cycles'] > 0:
            self.logger.info(f"   ✅ Stock accuracy: {stats['recent_stock_accuracy']:.1%}")
            self.logger.info(f"   ✅ Crypto accuracy: {stats['recent_crypto_accuracy']:.1%}")
        
        if stats['is_ready_for_paper_trading']:
            self.logger.info("   🚀 STATUS: READY for paper trading!")
        else:
            cycles_needed = max(0, 200 - stats['total_cycles'])
            self.logger.info(f"   📚 STATUS: Learning phase - need {cycles_needed} more cycles for readiness")
        
        self.logger.info("")
        
        # Show current status
        current_mode = self.determine_learning_mode()
        next_event, next_time = self.get_next_market_event()
        
        self.logger.info("🎯 Mode: Automatic switching based on market hours")
        self.logger.info(f"🔄 Current mode: {current_mode.title()}")
        self.logger.info(f"📅 Next switch: {next_event} at {next_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        self.logger.info("=" * 60)
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Run learning cycle
                self.learning_cycle()
                
                # Dynamic interval based on mode
                if self.current_mode == "crypto":
                    # Crypto is 24/7, can afford longer intervals
                    interval = base_interval_minutes * 2
                    self.logger.info(f"[WAIT] Crypto mode - waiting {interval} minutes for next cycle...")
                else:
                    # Stocks during market hours, shorter intervals
                    interval = base_interval_minutes
                    self.logger.info(f"[WAIT] Stock mode - waiting {interval} minutes for next cycle...")
                
                # Wait with early exit check for mode changes
                for i in range(interval * 60):  # Convert to seconds
                    if not self.is_running:
                        break
                    
                    # Check for mode change every minute
                    if i % 60 == 0:
                        new_mode = self.determine_learning_mode()
                        if new_mode != self.current_mode:
                            self.logger.info(f"[MODE CHANGE] Switching to {new_mode} mode early!")
                            break
                    
                    time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("[STOP] Smart learning stopped by user")
        finally:
            self.stop_learning()
    
    def stop_learning(self):
        """Stop learning and save final data with backup"""
        self.is_running = False
        
        # Save with backup
        self.save_learning_data(backup=True)
        
        # Get final statistics
        stats = self.get_learning_statistics()
        
        self.logger.info("[STOP] Persistent Smart ML learning system stopped")
        self.logger.info("=" * 60)
        self.logger.info(f"[FINAL STATS] Total learning cycles: {stats['total_cycles']}")
        self.logger.info(f"[FINAL STATS] Stock cycles: {stats['stock_cycles']}")
        self.logger.info(f"[FINAL STATS] Crypto cycles: {stats['crypto_cycles']}")
        self.logger.info(f"[FINAL STATS] Stock data points: {stats['stock_data_points']}")
        self.logger.info(f"[FINAL STATS] Crypto data points: {stats['crypto_data_points']}")
        self.logger.info(f"[FINAL STATS] Learning maturity: {stats['learning_maturity']:.1%}")
        self.logger.info(f"[FINAL STATS] Total runtime: {stats['total_runtime_hours']:.1f} hours")
        
        if stats['is_ready_for_paper_trading']:
            self.logger.info("🚀 [READY] System is ready for paper trading!")
        else:
            cycles_needed = max(0, 200 - stats['total_cycles'])
            self.logger.info(f"📚 [LEARNING] Need {cycles_needed} more cycles for paper trading readiness")
        
        self.logger.info(f"💾 [PERSISTENCE] Learning data saved for next session")
        self.logger.info("=" * 60)

def main():
    # Initialize learner before any prompts
    learner = SmartMLLearner()
    # Option to run historic training
    hist_train = input("\nRun historic training? (y/N): ").strip().lower()
    if hist_train == 'y':
        mode = input("Train on stocks or crypto? (stocks/crypto): ").strip().lower()
        if mode not in ['stocks', 'crypto']:
            print("Invalid mode. Defaulting to 'stocks'.")
            mode = 'stocks'
        print("Select lookback period:")
        print("  1) 30 days\n  2) 90 days\n  3) 180 days\n  4) 1 year\n  5) 2 years")
        lookback_choice = input("Enter choice (1-5): ").strip()
        lookback_map = {
            '1': 30,
            '2': 90,
            '3': 180,
            '4': 365,
            '5': 730
        }
        lookback_days = lookback_map.get(lookback_choice, 30)
        from datetime import date
        end_date = datetime.now().date()
        # For stocks, set end_date to last completed market day (weekday, not today if today is not finished)
        if mode == 'stocks':
            # If today is Saturday/Sunday, go back to Friday
            while end_date.weekday() > 4:  # 0=Mon, 4=Fri
                end_date -= timedelta(days=1)
            # If today is a weekday but before 4:30pm ET, use previous weekday
            now_et = datetime.now(pytz.timezone('US/Eastern'))
            if end_date == now_et.date() and now_et.hour < 16 or (now_et.hour == 16 and now_et.minute < 30):
                end_date -= timedelta(days=1)
                while end_date.weekday() > 4:
                    end_date -= timedelta(days=1)
            # Count back only trading days
            trading_days = pd.bdate_range(end=end_date, periods=lookback_days)
            start_date = trading_days[0].date()
        else:
            start_date = end_date - timedelta(days=lookback_days)
        # Choose interval based on lookback
        if lookback_days <= 60:
            interval = '10m' if mode == 'crypto' else '5m'
        else:
            interval = '1h'
        print(f"Training from {start_date} to {end_date} ({lookback_days} days) with interval {interval}")
        learner.train_on_historic_data(str(start_date), str(end_date), mode, interval=interval)
        print("\n✅ Historic training complete. Exiting.")
        return
    """Main function"""
    print("🤖 Persistent Smart ML Learning System")
    print("=" * 40)
    print("Intelligent market-hours aware learning:")
    print("📈 Market Hours (9:30 AM - 4:00 PM ET): Learn from STOCKS")
    print("🪙 After Hours & Weekends: Learn from CRYPTO")
    print("🔄 Automatic switching based on market schedule")
    print("📊 Cross-mode pattern analysis")
    print("💾 Persistent learning across restarts")
    print()
    
    # Initialize learner to show current stats
    learner = SmartMLLearner()
    stats = learner.get_learning_statistics()
    
    if stats['total_cycles'] > 0:
        print("📈 CURRENT LEARNING STATUS:")
        print(f"   🔄 Total cycles: {stats['total_cycles']}")
        print(f"   📊 Stock cycles: {stats['stock_cycles']}")
        print(f"   🪙 Crypto cycles: {stats['crypto_cycles']}")
        print(f"   📦 Data points: {stats['total_data_points']}")
        print(f"   🎯 Learning maturity: {stats['learning_maturity']:.1%}")
        print(f"   📈 Readiness: {stats['readiness_percentage']:.0f}%")
        
        if stats['is_ready_for_paper_trading']:
            print("   🚀 STATUS: READY for paper trading!")
        else:
            cycles_needed = max(0, 200 - stats['total_cycles'])
            print(f"   📚 STATUS: Need {cycles_needed} more cycles for readiness")
        print()
    
    # Configuration
    interval = input("Enter base interval in minutes (default 15): ").strip()
    try:
        interval = int(interval) if interval else 15
    except ValueError:
        interval = 15
    
    print(f"\\n⚙️  Configuration:")
    print(f"   📊 Stocks: 14 symbols (SPY, QQQ, AMD, NVDA, etc.)")
    print(f"   🪙 Crypto: 15 symbols (BTC, ETH, BNB, ADA, etc.)")
    print(f"   ⏰ Base interval: {interval} minutes")
    print(f"   🔄 Auto mode switching: Enabled")
    print(f"   💾 Persistent learning: Enabled")
    
    confirm = input("\\nStart smart learning? (y/N): ").strip().lower()
    
    if confirm == 'y':
        try:
            learner.start_smart_learning(base_interval_minutes=interval)
        except KeyboardInterrupt:
            print("\\n⚠️  Learning interrupted by user")
            learner.stop_learning()
    else:
        print("❌ Learning cancelled")

if __name__ == "__main__":
    main()
