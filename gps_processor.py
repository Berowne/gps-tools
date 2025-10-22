import os
import sys
import argparse
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, time as dtime, timedelta, timezone
import gpxpy
import gpxpy.gpx
from fastkml import kml
from pykml import parser as kml_parser
from haversine import haversine, Unit
import json

class GPSDataProcessor:
    """
    A class to process various GPS data file formats including GPX, KML, and others.
    """
    
    def __init__(self, input_dir: str, time_range: Optional[Tuple[dtime, dtime]] = None, tzinfo: Optional[timezone] = None, speed_unit: str = 'knots', max_accel_ms2: float = 5.0):
        """
        Initialize the GPSDataProcessor with the input directory containing GPS files.
        
        Args:
            input_dir (str): Path to the directory containing GPS files
            time_range (Optional[Tuple[datetime.time, datetime.time]]): Local time window filter (start, end)
            tzinfo (Optional[datetime.timezone]): Timezone for interpreting local times
        """
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {input_dir}")
        self.time_range = time_range
        self.tzinfo = tzinfo or datetime.now().astimezone().tzinfo
        self.speed_unit = speed_unit.lower()
        self.max_accel_ms2 = float(max_accel_ms2)
    
    def get_gps_files(self) -> List[Path]:
        """
        Get a list of all supported GPS files in the input directory.
        
        Returns:
            List[Path]: List of Path objects for supported GPS files
        """
        supported_extensions = ['.gpx', '.kml', '.kmz', '.fit', '.tcx', '.oao']
        gps_files = []
        
        for ext in supported_extensions:
            gps_files.extend(self.input_dir.glob(f'*{ext}'))
        
        return gps_files
    
    def read_gpx(self, file_path: Path) -> Dict[str, Any]:
        """Read and parse a GPX file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            gpx = gpxpy.parse(f)
        
        data = {
            'filename': file_path.name,
            'type': 'gpx',
            'waypoints': [],
            'tracks': [],
            'routes': []
        }
        
        # Extract waypoints
        for waypoint in gpx.waypoints:
            data['waypoints'].append({
                'latitude': waypoint.latitude,
                'longitude': waypoint.longitude,
                'elevation': waypoint.elevation,
                'time': waypoint.time.isoformat() if waypoint.time else None,
                'name': waypoint.name or '',
                'description': waypoint.description or ''
            })
        
        # Extract tracks
        for track in gpx.tracks:
            track_data = {
                'name': track.name or '',
                'description': track.description or '',
                'segments': []
            }
            
            for segment in track.segments:
                segment_data = []
                for point in segment.points:
                    segment_data.append({
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation,
                        'time': point.time.isoformat() if point.time else None
                    })
                track_data['segments'].append(segment_data)
            
            data['tracks'].append(track_data)
        
        return data
    
    def read_kml(self, file_path: Path) -> Dict[str, Any]:
        """Read and parse a KML file using pykml, extracting gx:Track points if present."""
        with open(file_path, 'r', encoding='utf-8') as f:
            xml = f.read()

        root = kml_parser.fromstring(xml)

        KML_NS = 'http://www.opengis.net/kml/2.2'
        GX_NS = 'http://www.google.com/kml/ext/2.2'
        ns = { 'kml': KML_NS, 'gx': GX_NS }

        data: Dict[str, Any] = {
            'filename': file_path.name,
            'type': 'kml',
            'waypoints': [],
            'tracks': [],
            'routes': [],
            'features': []
        }

        # 1) Try to extract gx:Track (paired <when> and <gx:coord>)
        placemarks = root.xpath('.//kml:Placemark', namespaces=ns)
        for pm in placemarks:
            tracks = pm.xpath('.//gx:Track', namespaces=ns)
            for trk in tracks:
                whens = trk.xpath('./kml:when', namespaces=ns)
                coords = trk.xpath('./gx:coord', namespaces=ns)
                # Pair by index
                count = min(len(whens), len(coords))
                if count == 0:
                    continue
                seg_points: List[Dict[str, Any]] = []
                for i in range(count):
                    t = whens[i].text.strip() if whens[i].text else None
                    coord_txt = coords[i].text.strip() if coords[i].text else ''
                    # gx:coord: lon lat alt
                    try:
                        lon, lat, alt = [float(x) for x in coord_txt.split()] + [0.0, 0.0, 0.0]
                        lon, lat, alt = lon, lat, alt
                    except Exception:
                        # Fallback try lon,lat,alt CSV (non-standard)
                        parts = coord_txt.replace(',', ' ').split()
                        if len(parts) >= 2:
                            lon = float(parts[0]); lat = float(parts[1])
                            alt = float(parts[2]) if len(parts) >= 3 else None
                        else:
                            continue
                    seg_points.append({
                        'latitude': lat,
                        'longitude': lon,
                        'elevation': alt,
                        'time': t
                    })
                if seg_points:
                    pm_name = self._kml_text(pm, './kml:name', ns)
                    pm_desc = self._kml_text(pm, './kml:description', ns)
                    data['tracks'].append({
                        'name': pm_name,
                        'description': pm_desc,
                        'segments': [seg_points]
                    })

        # 2) If no gx:Track, capture simple geometry as features for visibility
        if not data['tracks']:
            # LineString coordinates (no timestamps => no metrics)
            lines = root.xpath('.//kml:LineString/kml:coordinates', namespaces=ns)
            for ln in lines:
                coords_text = ln.text or ''
                coords_list = []
                for triplet in coords_text.strip().split():
                    parts = triplet.split(',')
                    if len(parts) >= 2:
                        lon = float(parts[0]); lat = float(parts[1])
                        coords_list.append((lon, lat))
                data['features'].append({
                    'name': '',
                    'description': '',
                    'geometry_type': 'LineString',
                    'coordinates': coords_list
                })

        return data
    
    def process_all_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Process all supported GPS files in the input directory.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping filenames to their parsed data with metrics
        """
        gps_files = self.get_gps_files()
        results: Dict[str, Dict[str, Any]] = {}
        
        for file_path in gps_files:
            try:
                if file_path.suffix.lower() == '.gpx':
                    parsed = self.read_gpx(file_path)
                    metrics = self.compute_metrics_for_gpx(parsed)
                    parsed['metrics'] = metrics
                    results[file_path.name] = parsed
                elif file_path.suffix.lower() in ['.kml', '.kmz']:
                    parsed = self.read_kml(file_path)
                    # If KML has gx:Track points, compute metrics using same pipeline
                    if parsed.get('tracks'):
                        metrics = self.compute_metrics_for_gpx(parsed)
                        parsed['metrics'] = metrics
                    else:
                        parsed['metrics'] = {}
                    results[file_path.name] = parsed
                elif file_path.suffix.lower() == '.oao':
                    parsed = self.read_oao(file_path)
                    if parsed.get('tracks'):
                        metrics = self.compute_metrics_for_gpx(parsed)
                        parsed['metrics'] = metrics
                    else:
                        parsed['metrics'] = {}
                    results[file_path.name] = parsed
                else:
                    print(f"Skipping unsupported file format: {file_path}", file=sys.stderr)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}", file=sys.stderr)
        
        return results

    # ===== Metric helpers =====
    def _collect_points(self, gpx_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        for track in gpx_data.get('tracks', []):
            for seg in track.get('segments', []):
                for p in seg:
                    if p.get('time'):
                        points.append(p)
        # Sort by time
        points.sort(key=lambda x: x['time'])
        return points

    def _kml_text(self, node, xpath_expr: str, namespaces: Dict[str, str]) -> str:
        """Extract first text value from an XPath; returns '' if missing."""
        try:
            vals = node.xpath(xpath_expr, namespaces=namespaces)
            if not vals:
                return ''
            # pykml may return lxml elements or strings
            v = vals[0]
            if hasattr(v, 'text'):
                return (v.text or '').strip()
            return str(v).strip()
        except Exception:
            return ''

    def _parse_dt(self, iso_str: str) -> Optional[datetime]:
        try:
            # fromisoformat supports offsets
            return datetime.fromisoformat(iso_str)
        except Exception:
            return None

    def _filter_by_time_window(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.time_range:
            return points
        start_t, end_t = self.time_range
        filtered: List[Dict[str, Any]] = []
        for p in points:
            ts = self._parse_dt(p['time'])
            if not ts:
                continue
            # Ensure timezone-aware
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            local_ts = ts.astimezone(self.tzinfo)
            lt = local_ts.timetz().replace(tzinfo=None)
            in_range = False
            if start_t <= end_t:
                in_range = (lt >= start_t and lt <= end_t)
            else:
                # window crosses midnight
                in_range = (lt >= start_t or lt <= end_t)
            if in_range:
                filtered.append(p)
        return filtered

    def _pair_distance_m(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        return haversine((a['latitude'], a['longitude']), (b['latitude'], b['longitude']), unit=Unit.METERS)

    def _pair_time_s(self, a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
        ta = self._parse_dt(a['time']) if isinstance(a.get('time'), str) else a.get('time')
        tb = self._parse_dt(b['time']) if isinstance(b.get('time'), str) else b.get('time')
        if not ta or not tb:
            return None
        return (tb - ta).total_seconds()

    def _filter_by_acceleration(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(points) < 3:
            return points
        kept: List[Dict[str, Any]] = [points[0]]
        for i in range(1, len(points) - 1):
            p0 = kept[-1]
            p1 = points[i]
            p2 = points[i + 1]
            dt1 = self._pair_time_s(p0, p1) or 0.0
            dt2 = self._pair_time_s(p1, p2) or 0.0
            if dt1 <= 0 or dt2 <= 0:
                kept.append(p1)
                continue
            d1 = self._pair_distance_m(p0, p1)
            d2 = self._pair_distance_m(p1, p2)
            v1 = d1 / dt1
            v2 = d2 / dt2
            dt_mean = 0.5 * (dt1 + dt2)
            if dt_mean <= 0:
                kept.append(p1)
                continue
            a = (v2 - v1) / dt_mean
            if abs(a) <= self.max_accel_ms2:
                kept.append(p1)
            else:
                continue
        kept.append(points[-1])
        return kept

    # ===== Units helpers =====
    def _kmh_to_unit(self, kmh: float) -> float:
        u = self.speed_unit
        if u == 'kmh' or u == 'km/h':
            return kmh
        if u == 'knots' or u == 'kt' or u == 'kts':
            return kmh / 1.852
        if u == 'mph':
            return kmh * 0.621371
        if u in ('ms', 'm/s'):
            return kmh / 3.6
        # default
        return kmh

    def _unit_label(self) -> str:
        u = self.speed_unit
        if u in ('kmh', 'km/h'):
            return 'km/h'
        if u in ('knots', 'kt', 'kts'):
            return 'knots'
        if u == 'mph':
            return 'mph'
        if u in ('ms', 'm/s'):
            return 'm/s'
        return 'km/h'

    def compute_total_distance(self, points: List[Dict[str, Any]]) -> float:
        d = 0.0
        for i in range(1, len(points)):
            d += self._pair_distance_m(points[i-1], points[i])
        return d

    def compute_max_avg_speed_over_time_window(self, points: List[Dict[str, Any]], window_seconds: int) -> float:
        # Sliding window by time
        if not points:
            return 0.0
        i = 0
        max_kmh = 0.0
        # Precompute distances between consecutive points
        seg_d = [0.0] + [self._pair_distance_m(points[j-1], points[j]) for j in range(1, len(points))]
        j = 0
        while i < len(points):
            # Move j to ensure window >= window_seconds
            while j < len(points) and (self._pair_time_s(points[i], points[j]) or 0) < window_seconds:
                j += 1
            if j >= len(points):
                break
            # Compute distance from i to j
            dist = 0.0
            for k in range(i+1, j+1):
                dist += seg_d[k]
            dt = self._pair_time_s(points[i], points[j]) or 0.0
            if dt > 0:
                kmh = (dist / dt) * 3.6
                if kmh > max_kmh:
                    max_kmh = kmh
            i += 1
        return max_kmh

    def compute_top_n_avg_speeds_over_time_window(self, points: List[Dict[str, Any]], window_seconds: int, top_n: int) -> List[float]:
        scores: List[float] = []
        if not points:
            return scores
        i = 0
        seg_d = [0.0] + [self._pair_distance_m(points[j-1], points[j]) for j in range(1, len(points))]
        j = 0
        while i < len(points):
            while j < len(points) and (self._pair_time_s(points[i], points[j]) or 0) < window_seconds:
                j += 1
            if j >= len(points):
                break
            dist = 0.0
            for k in range(i+1, j+1):
                dist += seg_d[k]
            dt = self._pair_time_s(points[i], points[j]) or 0.0
            if dt > 0:
                kmh = (dist / dt) * 3.6
                scores.append(kmh)
            i += 1
        scores.sort(reverse=True)
        return scores[:top_n]

    def compute_best_speed_over_distance(self, points: List[Dict[str, Any]], target_m: float) -> float:
        # Two-pointer window by distance
        if not points:
            return 0.0
        seg_d = [0.0] + [self._pair_distance_m(points[j-1], points[j]) for j in range(1, len(points))]
        prefix = [0.0]
        for k in range(1, len(points)):
            prefix.append(prefix[-1] + seg_d[k])
        best = 0.0
        j = 1
        for i in range(len(points)):
            # Advance j until distance >= target
            while j < len(points) and (prefix[j] - prefix[i]) < target_m:
                j += 1
            if j >= len(points):
                break
            dist = prefix[j] - prefix[i]
            dt = self._pair_time_s(points[i], points[j]) or 0.0
            if dt > 0:
                kmh = (dist / dt) * 3.6
                if kmh > best:
                    best = kmh
        return best

    def compute_average_speed_over_valid_windows(self, points: List[Dict[str, Any]], window_seconds: int = 10, min_kmh: float = 4.0) -> float:
        speeds = self.compute_top_n_avg_speeds_over_time_window(points, window_seconds, top_n=10_000)
        valid = [s for s in speeds if s >= min_kmh]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)

    def compute_alpha500_speed(self, points: List[Dict[str, Any]], loop_radius_m: float = 50.0, target_path_m: float = 500.0) -> float:
        # Search segments whose cumulative path length >= target_path_m and endpoint within loop_radius of start
        if not points:
            return 0.0
        seg_d = [0.0] + [self._pair_distance_m(points[j-1], points[j]) for j in range(1, len(points))]
        prefix = [0.0]
        for k in range(1, len(points)):
            prefix.append(prefix[-1] + seg_d[k])
        best = 0.0
        j = 1
        for i in range(len(points)):
            # expand j until path >= target_path_m
            while j < len(points) and (prefix[j] - prefix[i]) < target_path_m:
                j += 1
            if j >= len(points):
                break
            # Check proximity condition
            if self._pair_distance_m(points[i], points[j]) <= loop_radius_m:
                path = prefix[j] - prefix[i]
                dt = self._pair_time_s(points[i], points[j]) or 0.0
                if dt > 0:
                    kmh = (path / dt) * 3.6
                    if kmh > best:
                        best = kmh
        return best

    def compute_metrics_for_gpx(self, gpx_data: Dict[str, Any]) -> Dict[str, Any]:
        # Flatten points and apply time window
        pts = self._collect_points(gpx_data)
        pts = self._filter_by_time_window(pts)
        pts = self._filter_by_acceleration(pts)
        if len(pts) < 2:
            return {
                'total_distance_m': 0.0,
                'max_2s_kmh': 0.0,
                'top5_10s_kmh': [],
                'best_100m_kmh': 0.0,
                'best_500m_kmh': 0.0,
                'best_1nm_kmh': 0.0,
                'avg_speed_kmh': 0.0,
                'alpha500_kmh': 0.0,
            }
        total_m = self.compute_total_distance(pts)
        max_2s = self.compute_max_avg_speed_over_time_window(pts, 2)
        top5_10s = self.compute_top_n_avg_speeds_over_time_window(pts, 10, 5)
        best_100m = self.compute_best_speed_over_distance(pts, 100.0)
        best_500m = self.compute_best_speed_over_distance(pts, 500.0)
        best_1nm = self.compute_best_speed_over_distance(pts, 1852.0)
        avg_speed = self.compute_average_speed_over_valid_windows(pts, 10, 4.0)
        alpha500 = self.compute_alpha500_speed(pts, 50.0, 500.0)
        return {
            'total_distance_m': total_m,
            'max_2s_kmh': max_2s,
            'top5_10s_kmh': top5_10s,
            'best_100m_kmh': best_100m,
            'best_500m_kmh': best_500m,
            'best_1nm_kmh': best_1nm,
            'avg_speed_kmh': avg_speed,
            'alpha500_kmh': alpha500,
        }

    def read_oao(self, file_path: Path) -> Dict[str, Any]:
        """Try to read an .oao file by sniffing its format (GPX/KML/JSON).
        Returns the internal common structure with tracks/segments/points.
        """
        # Sniff header
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                head = f.read(2048)
        except UnicodeDecodeError:
            # Binary or different encoding not supported yet
            raise ValueError(f"Unsupported encoding in OAO file: {file_path}")

        lower = head.lower()
        if '<gpx' in lower:
            return self.read_gpx(file_path)
        if '<kml' in lower or '<gx:track' in lower:
            return self.read_kml(file_path)

        # Try JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
        except Exception:
            # Unknown OAO format
            return {
                'filename': file_path.name,
                'type': 'oao',
                'waypoints': [],
                'tracks': [],
                'routes': [],
                'raw': None,
            }

        def coerce_point(pt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            lat = pt.get('lat') if 'lat' in pt else pt.get('latitude')
            lon = pt.get('lon') if 'lon' in pt else pt.get('longitude')
            t = pt.get('time') or pt.get('timestamp') or pt.get('datetime')
            if lat is None or lon is None or t is None:
                return None
            elev = pt.get('ele') if 'ele' in pt else pt.get('elevation')
            return {
                'latitude': float(lat),
                'longitude': float(lon),
                'elevation': float(elev) if elev is not None else None,
                'time': str(t),
            }

        tracks: List[Dict[str, Any]] = []

        if isinstance(obj, list):
            # Assume list of points
            pts: List[Dict[str, Any]] = []
            for item in obj:
                if isinstance(item, dict):
                    cp = coerce_point(item)
                    if cp:
                        pts.append(cp)
            if pts:
                tracks.append({'name': '', 'description': '', 'segments': [pts]})
        elif isinstance(obj, dict):
            if 'tracks' in obj:
                # Try to normalize tracks->segments->points
                for tr in obj.get('tracks', []):
                    segs_norm: List[List[Dict[str, Any]]] = []
                    segs = tr.get('segments') or tr.get('segments_points') or tr.get('points')
                    if isinstance(segs, list):
                        # If segments is a list of lists
                        if segs and isinstance(segs[0], list):
                            for seg in segs:
                                seg_pts: List[Dict[str, Any]] = []
                                for pt in seg:
                                    if isinstance(pt, dict):
                                        cp = coerce_point(pt)
                                        if cp:
                                            seg_pts.append(cp)
                                if seg_pts:
                                    segs_norm.append(seg_pts)
                        else:
                            # segments is a flat list of points
                            seg_pts = []
                            for pt in segs:
                                if isinstance(pt, dict):
                                    cp = coerce_point(pt)
                                    if cp:
                                        seg_pts.append(cp)
                            if seg_pts:
                                segs_norm.append(seg_pts)
                    if segs_norm:
                        tracks.append({
                            'name': tr.get('name', ''),
                            'description': tr.get('description', ''),
                            'segments': segs_norm
                        })
            elif 'points' in obj and isinstance(obj['points'], list):
                seg_pts: List[Dict[str, Any]] = []
                for pt in obj['points']:
                    if isinstance(pt, dict):
                        cp = coerce_point(pt)
                        if cp:
                            seg_pts.append(cp)
                if seg_pts:
                    tracks.append({'name': obj.get('name', ''), 'description': obj.get('description', ''), 'segments': [seg_pts]})

        return {
            'filename': file_path.name,
            'type': 'oao',
            'waypoints': [],
            'tracks': tracks,
            'routes': [],
        }

    def results_to_html(self, results: Dict[str, Dict[str, Any]]) -> str:
        # Build a sortable HTML table and include time filter/data span subtitle
        rows = []
        unit_lbl = self._unit_label()

        # Compute overall data time span
        min_ts = None
        max_ts = None
        for data in results.values():
            for track in data.get('tracks', []):
                for seg in track.get('segments', []):
                    for p in seg:
                        t = p.get('time')
                        if not t:
                            continue
                        dt = self._parse_dt(t) if isinstance(t, str) else t
                        if not dt:
                            continue
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        lt = dt.astimezone(self.tzinfo)
                        if min_ts is None or lt < min_ts:
                            min_ts = lt
                        if max_ts is None or lt > max_ts:
                            max_ts = lt

        time_filter_text = "All times"
        if self.time_range:
            st, et = self.time_range
            # Format HH:MM using zero-pad
            time_filter_text = f"Local time filter: {st.strftime('%H:%M')}-{et.strftime('%H:%M')} ({self.tzinfo.tzname(None) or 'Local TZ'})"
        span_text = ""
        if min_ts and max_ts:
            span_text = f"Data time span: {min_ts.strftime('%Y-%m-%d %H:%M')} — {max_ts.strftime('%Y-%m-%d %H:%M')} ({self.tzinfo.tzname(None) or 'Local TZ'})"
        subtitle = f"{time_filter_text}" + (f" &nbsp; | &nbsp; {span_text}" if span_text else "")

        header = (
            "<tr>"
            "<th title='Row number (fixed)'>#</th>"
            "<th>File</th>"
            "<th>Total Distance (km)</th>"
            f"<th>Max 2s ({unit_lbl})</th>"
            f"<th>Top 5 x 10s ({unit_lbl})</th>"
            f"<th>100 m ({unit_lbl})</th>"
            f"<th>500 m ({unit_lbl})</th>"
            f"<th>1 NM ({unit_lbl})</th>"
            f"<th>Avg >=4 km/h ({unit_lbl})</th>"
            f"<th>Alpha 500 ({unit_lbl})</th>"
            "</tr>"
        )
        idx = 1
        for fname, data in results.items():
            m = data.get('metrics', {})
            top5 = ", ".join(f"{self._kmh_to_unit(v):.2f}" for v in m.get('top5_10s_kmh', [])) if m else ""
            row = (
                f"<tr>"
                f"<td class='rownum' data-index='{idx}'>{idx}</td>"
                f"<td>{fname}</td>"
                f"<td data-num='1'>{(m.get('total_distance_m', 0.0)/1000):.3f}</td>"
                f"<td data-num='1'>{self._kmh_to_unit(m.get('max_2s_kmh', 0.0)):.2f}</td>"
                f"<td>{top5}</td>"
                f"<td data-num='1'>{self._kmh_to_unit(m.get('best_100m_kmh', 0.0)):.2f}</td>"
                f"<td data-num='1'>{self._kmh_to_unit(m.get('best_500m_kmh', 0.0)):.2f}</td>"
                f"<td data-num='1'>{self._kmh_to_unit(m.get('best_1nm_kmh', 0.0)):.2f}</td>"
                f"<td data-num='1'>{self._kmh_to_unit(m.get('avg_speed_kmh', 0.0)):.2f}</td>"
                f"<td data-num='1'>{self._kmh_to_unit(m.get('alpha500_kmh', 0.0)):.2f}</td>"
                f"</tr>"
            )
            rows.append(row)
            idx += 1
        html = (
            "<!DOCTYPE html>\n<html><head><meta charset='utf-8'><title>GPS Metrics</title>"
            "<style>body{font-family:Arial,sans-serif;padding:20px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:8px;text-align:center;}th{background:#f4f4f4;position:sticky;top:0;cursor:pointer;}tr:nth-child(even){background:#fafafa;}caption{caption-side:top;text-align:left;margin-bottom:10px;color:#444;}small.sub{display:block;margin-top:4px;color:#666;}td.rownum, th:first-child{width:60px;cursor:default;}</style>"
            "<script>\n"
            "function renumber(){const tbody=document.getElementById('metrics').tBodies[0];Array.from(tbody.rows).forEach((r,i)=>{r.cells[0].textContent=String(i+1);});}\n"
            "function sortTable(n){if(n===0){return;}const table=document.getElementById('metrics');const tbody=table.tBodies[0];let rows=Array.from(tbody.rows);let dir=table.getAttribute('data-sort-dir')==='asc'?'desc':'asc';const isNum=(cell)=>cell.hasAttribute('data-num');rows.sort((a,b)=>{const A=a.cells[n].innerText.trim();const B=b.cells[n].innerText.trim();if(isNum(a.cells[n])||isNum(b.cells[n])){const x=parseFloat(A)||0;const y=parseFloat(B)||0;return dir==='asc'?x-y:y-x;}return dir==='asc'?A.localeCompare(B):B.localeCompare(A);});tbody.innerHTML='';rows.forEach(r=>tbody.appendChild(r));table.setAttribute('data-sort-dir',dir);renumber();}\n"
            "window.addEventListener('DOMContentLoaded',()=>{document.querySelectorAll('#metrics thead th').forEach((th,i)=>{th.addEventListener('click',()=>sortTable(i));});renumber();});\n"
            "</script>"
            "</head><body>"
            f"<h2>GPS Metrics<small class='sub'>{subtitle}</small></h2>"
            "<table id='metrics' data-sort-dir='asc'>"
            "<thead>"
            f"{header}"
            "</thead><tbody>"
            f"{''.join(rows)}"
            "</tbody></table>"
            "</body></html>"
        )
        return html

def parse_time_range(spec: Optional[str]) -> Optional[Tuple[dtime, dtime]]:
    if not spec:
        return None
    try:
        left, right = spec.split('-')
        h1, m1 = [int(x) for x in left.split(':')]
        h2, m2 = [int(x) for x in right.split(':')]
        return dtime(hour=h1, minute=m1), dtime(hour=h2, minute=m2)
    except Exception:
        raise ValueError("--time-range must be in HH:MM-HH:MM format, e.g., 12:34-13:45")


def parse_timezone(spec: Optional[str]) -> timezone:
    if not spec:
        return datetime.now().astimezone().tzinfo or timezone.utc
    s = spec.strip().upper().replace('UTC', 'GMT')
    if not s.startswith('GMT'):
        raise ValueError("--timezone must look like GMT+11 or UTC-2")
    off = s[3:]
    try:
        sign = 1
        if off.startswith('+'):
            sign = 1
            off = off[1:]
        elif off.startswith('-'):
            sign = -1
            off = off[1:]
        if ':' in off:
            hh, mm = off.split(':')
            hours = int(hh)
            minutes = int(mm)
        else:
            hours = int(off)
            minutes = 0
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    except Exception:
        raise ValueError("Invalid timezone offset. Use e.g., GMT+11 or UTC+10:30")


def main():
    parser = argparse.ArgumentParser(description="Process GPS files and compute metrics.")
    parser.add_argument('directory', help='Path to directory containing GPS files')
    parser.add_argument('--time-range', dest='time_range', help='Local time range filter, e.g., 12:34-13:45')
    parser.add_argument('--timezone', dest='tz', help='Timezone like GMT+11 or UTC-2. Defaults to system tz')
    parser.add_argument('--html', dest='html', default=None, help='Output HTML file path (default: <directory>/gps_metrics.html)')
    parser.add_argument('--units', dest='units', default='knots', choices=['knots','kmh','km/h','mph','ms','m/s'], help='Units for speed metrics (default: knots)')
    parser.add_argument('--json', dest='json', default='gps_data_summary.json', help='Output JSON file path')
    parser.add_argument('--accel-max', dest='accel_max', type=float, default=5.0, help='Maximum allowed acceleration (m/s^2) for outlier filtering')
    args = parser.parse_args()

    try:
        tr = parse_time_range(args.time_range)
        tzinfo = parse_timezone(args.tz)
        processor = GPSDataProcessor(args.directory, time_range=tr, tzinfo=tzinfo, speed_unit=args.units, max_accel_ms2=args.accel_max)
        results = processor.process_all_files()

        # Save JSON
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {args.json}")

        # Save HTML
        html = processor.results_to_html(results)
        if args.html:
            html_out = args.html
        else:
            # Derive earliest local date from data
            min_ts = None
            for data in results.values():
                for track in data.get('tracks', []):
                    for seg in track.get('segments', []):
                        for p in seg:
                            t = p.get('time')
                            if not t:
                                continue
                            dt = processor._parse_dt(t) if isinstance(t, str) else t
                            if not dt:
                                continue
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            lt = dt.astimezone(processor.tzinfo)
                            if min_ts is None or lt < min_ts:
                                min_ts = lt
            date_part = f"_{min_ts.strftime('%Y%m%d')}" if min_ts else ''
            # Default filename, optionally suffixed with time-range if provided
            time_part = ''
            if tr:
                st, et = tr
                time_part = f"_{st.strftime('%H%M')}-{et.strftime('%H%M')}"
            html_out = str((Path(args.directory) / f"gps_metrics{date_part}{time_part}.html").resolve())
        with open(html_out, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Saved HTML: {html_out}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
