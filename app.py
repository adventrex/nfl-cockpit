# Load Current Season (2025) - Try Harder with Fallback
    try:
        p = nfl.import_pbp_data([current_year], cache=False)
        w = nfl.import_weekly_data([current_year])
        
        if not p.empty:
            pbp_all.append(p)
            weekly_all.append(w)
            loaded_years.append(current_year)
            status_report[current_year] = "✅ Loaded (Library)"
        else:
            raise ValueError("Empty Library Result")
    except Exception as lib_err:
        # Library failed, try direct RAW GitHub access (Bleeding Edge)
        try:
            base_raw = "https://raw.githubusercontent.com/nflverse/nflverse-data/master/data"
            
            # 1. Try Play-by-Play
            url_p = f"{base_raw}/pbp/play_by_play_{current_year}.csv.gz"
            p_direct = pd.read_csv(url_p, compression='gzip', low_memory=False)
            
            # 2. Try Player Stats (Weekly)
            # Note: Raw path sometimes differs, try 'player_stats' folder
            url_w = f"{base_raw}/player_stats/player_stats_{current_year}.csv.gz"
            w_direct = pd.read_csv(url_w, compression='gzip', low_memory=False)

            if not p_direct.empty:
                pbp_all.append(p_direct)
                weekly_all.append(w_direct)
                loaded_years.append(current_year)
                status_report[current_year] = "✅ Loaded (Raw GitHub)"
            else:
                 raise ValueError("Raw Empty")
                 
        except Exception as raw_err:
            # 3. Fallback to Release URL
            try:
                url_w_rel = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{current_year}.csv.gz"
                w_rel = pd.read_csv(url_w_rel, compression='gzip', low_memory=False)
                
                url_p_rel = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{current_year}.csv.gz"
                p_rel = pd.read_csv(url_p_rel, compression='gzip', low_memory=False)
                
                if not p_rel.empty:
                    pbp_all.append(p_rel)
                    weekly_all.append(w_rel)
                    loaded_years.append(current_year)
                    status_report[current_year] = "✅ Loaded (Release URL)"
                else:
                    status_report[current_year] = "❌ Unavailable"
            except:
                status_report[current_year] = "❌ Unavailable"
