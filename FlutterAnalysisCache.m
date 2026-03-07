classdef FlutterAnalysisCache
    methods(Static)
        function [cache_loaded, lambda_F, plot_data] = try_load(results_mat_file, ~, force_resolve)
            cache_loaded = false;
            lambda_F = nan;
            plot_data = struct();

            if force_resolve || ~isfile(results_mat_file)
                return;
            end

            vars_in_file = {whos('-file', results_mat_file).name};
            if ~ismember('plot_data', vars_in_file)
                return;
            end

            if ismember('lambda_F', vars_in_file)
                S_lambda = load(results_mat_file, 'lambda_F');
                lambda_F = S_lambda.lambda_F;
            end

            S_plot = load(results_mat_file, 'plot_data');
            plot_data = S_plot.plot_data;
            cache_loaded = true;
        end

        function plot_data = save_with_plot_data(results_mat_file, params, solve_data)
            plot_data = FlutterAnalysisCache.extract_plot_data(params, solve_data);
            lambda_F = solve_data.lambda_F;

            save(results_mat_file, ...
                'params', 'lambda_F', 'plot_data', '-v7');
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
                plot_data.y_max_vm_steady, plot_data.max_vm_is_upper_steady] = ...
                FlutterAnalysisCache.extract_vm_window_peak( ...
                solve_data.vm_upper, solve_data.vm_lower, params, idx_steady);
        end

        function [vm_max, x_max, y_max, max_is_upper] = extract_vm_window_peak(vm_upper, vm_lower, params, idx_time)
            vm_upper_pt = squeeze(max(vm_upper(:, :, idx_time), [], 3));
            vm_lower_pt = squeeze(max(vm_lower(:, :, idx_time), [], 3));

            vm_pt_time = max(vm_upper_pt, vm_lower_pt);
            [vm_max, idx_pt] = max(vm_pt_time, [], 2);

            x_lin = linspace(0, params.a, params.disc_stress);
            y_lin = linspace(-params.b/2, params.b/2, params.disc_stress);
            [Xg, Yg] = meshgrid(x_lin, y_lin);
            x_points = reshape(Xg, 1, []);
            y_points = reshape(Yg, 1, []);

            x_max = x_points(idx_pt);
            y_max = y_points(idx_pt);

            upper_ge_lower = vm_upper_pt >= vm_lower_pt;
            idx_linear = sub2ind(size(upper_ge_lower), (1:size(upper_ge_lower, 1)).', idx_pt);
            max_is_upper = upper_ge_lower(idx_linear);
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
