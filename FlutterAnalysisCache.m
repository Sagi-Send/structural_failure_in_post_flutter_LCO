classdef FlutterAnalysisCache
    methods(Static)
        function [cache_loaded, lambda_F, plot_data] = try_load(results_mat_file, params, force_resolve)
            cache_loaded = false;
            lambda_F = nan;
            plot_data = struct();

            if force_resolve || ~isfile(results_mat_file)
                return;
            end

            vars_in_file = {whos('-file', results_mat_file).name};

            if ismember('lambda_F', vars_in_file)
                S_lambda = load(results_mat_file, 'lambda_F');
                lambda_F = S_lambda.lambda_F;
            end

            if ismember('plot_data', vars_in_file)
                S_plot = load(results_mat_file, 'plot_data');
                plot_data = S_plot.plot_data;

                if ~isfield(plot_data, 'max_vm_surface_sign_steady') && ...
                        all(ismember({'vm_upper', 'vm_lower'}, vars_in_file))
                    S_vm = load(results_mat_file, 'vm_upper', 'vm_lower');
                    Nt = numel(params.t_eval);
                    idx_steady = FlutterAnalysisCache.select_time_window_indices(Nt, 1-params.steady_frac, 1.0);
                    [plot_data.vm_max_steady, plot_data.x_max_vm_steady, ...
                        plot_data.y_max_vm_steady, plot_data.max_vm_surface_sign_steady] = ...
                        FlutterAnalysisCache.extract_vm_window_peak_with_surface( ...
                        S_vm.vm_upper, S_vm.vm_lower, params, idx_steady);
                end
            else
                S = load(results_mat_file, ...
                    'w_center', 'lco_amp', 'amp_steady', 'flutter_onset_idx', ...
                    'damping_array', 'natural_frequencies_hz_array', ...
                    'vm_upper', 'vm_lower', 'lambda_F');
                plot_data = FlutterAnalysisCache.extract_plot_data(params, S);
            end

            cache_loaded = true;
        end

        function plot_data = save_with_plot_data(results_mat_file, params, solve_data)
            plot_data = FlutterAnalysisCache.extract_plot_data(params, solve_data);

            w_center = solve_data.w_center;
            w_i = solve_data.w_i;
            lambda_F = solve_data.lambda_F;
            amp_steady = solve_data.amp_steady;
            lco_amp = amp_steady; % backward compatibility in cache file
            flutter_onset_idx = solve_data.flutter_onset_idx;
            natural_frequencies_hz_array = solve_data.natural_frequencies_hz_array;
            damping_array = solve_data.damping_array;
            unstable = solve_data.unstable;
            max_real_eig = solve_data.max_real_eig;
            vm_upper = solve_data.vm_upper;
            vm_lower = solve_data.vm_lower;

            save(results_mat_file, ...
                'params', 'w_center', 'w_i', 'lambda_F', 'lco_amp', ...
                'amp_steady', ...
                'flutter_onset_idx', 'natural_frequencies_hz_array', ...
                'damping_array', 'unstable', 'max_real_eig', ...
                'vm_upper', 'vm_lower', 'plot_data', '-v7.3');
        end

        function plot_data = extract_plot_data(params, solve_data)
            plot_data.lambda = params.lambda;
            plot_data.t_eval = params.t_eval;
            plot_data.pinf = params.pinf_sweep;
            plot_data.a = params.a;
            plot_data.b = params.b;
            plot_data.h = params.h;

            steady_frac = params.steady_frac;
            plot_data.w_center = solve_data.w_center;
            Nt = numel(params.t_eval);
            idx_steady = FlutterAnalysisCache.select_time_window_indices(Nt, 1-steady_frac, 1.0);

            if isfield(solve_data, 'amp_steady') && ~isempty(solve_data.amp_steady)
                plot_data.A_steady = solve_data.amp_steady;
            elseif isfield(solve_data, 'lco_amp') && ~isempty(solve_data.lco_amp)
                plot_data.A_steady = solve_data.lco_amp;
            else
                plot_data.A_steady = FlutterAnalysisCache.amplitude_from_window( ...
                    solve_data.w_center, idx_steady);
            end

            plot_data.lambda_F = solve_data.lambda_F;
            plot_data.flutter_idx = solve_data.flutter_onset_idx;
            plot_data.damping_array = solve_data.damping_array;
            plot_data.reduced_freq_array = ...
                FlutterAnalysisCache.nondimentionalize(params, solve_data.natural_frequencies_hz_array);

            [plot_data.vm_max_steady, plot_data.x_max_vm_steady, ...
                plot_data.y_max_vm_steady, plot_data.max_vm_surface_sign_steady] = ...
                FlutterAnalysisCache.extract_vm_window_peak_with_surface( ...
                solve_data.vm_upper, solve_data.vm_lower, params, idx_steady);
        end

        function [vm_max, x_max, y_max, surface_sign] = extract_vm_window_peak_with_surface(vm_upper, vm_lower, params, idx_time)
            vm_all = cat(4, vm_upper(:, :, idx_time), vm_lower(:, :, idx_time));
            n_pressures = size(vm_all, 1);
            n_points = size(vm_all, 2);
            n_time = size(vm_all, 3);

            vm_flat = reshape(vm_all, n_pressures, []);
            [vm_max, idx_flat] = max(vm_flat, [], 2);
            [idx_pt, ~, idx_surface] = ind2sub([n_points, n_time, 2], idx_flat);

            x_lin = linspace(0, params.a, params.disc_stress);
            y_lin = linspace(-params.b/2, params.b/2, params.disc_stress);
            [Xg, Yg] = meshgrid(x_lin, y_lin);
            x_points = reshape(Xg, 1, []);
            y_points = reshape(Yg, 1, []);

            x_max = x_points(idx_pt);
            y_max = y_points(idx_pt);
            surface_sign = 2 - idx_surface;
        end

        function amp = amplitude_from_window(w_center, idx_window)
            w_win = w_center(:, idx_window);
            amp = 0.5*(max(w_win, [], 2) - min(w_win, [], 2)).';
        end

        function idx_window = select_time_window_indices(Nt, startFrac, endFrac)
            i_start = max(1, floor(startFrac*Nt) + 1);
            i_end = min(Nt, max(i_start, floor(endFrac*Nt)));
            idx_window = i_start:i_end;
        end

        function reduced_freq_array = nondimentionalize(params, natural_frequencies_hz_array)
            Rgas = 287;
            a_inf = sqrt(params.gamma * Rgas * params.T0);
            Uinf = params.Minf * a_inf;
            Lref = params.a;

            reduced_freq_array = (2 * pi * natural_frequencies_hz_array) * (Lref / Uinf);
        end
    end
end
