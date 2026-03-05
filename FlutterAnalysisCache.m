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
            else
                S = load(results_mat_file, ...
                    'w_center', 'lco_amp', 'flutter_onset_idx', ...
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
            lco_amp = solve_data.lco_amp;
            flutter_onset_idx = solve_data.flutter_onset_idx;
            natural_frequencies_hz_array = solve_data.natural_frequencies_hz_array;
            damping_array = solve_data.damping_array;
            unstable = solve_data.unstable;
            max_real_eig = solve_data.max_real_eig;
            vm_upper = solve_data.vm_upper;
            vm_lower = solve_data.vm_lower;

            save(results_mat_file, ...
                'params', 'w_center', 'w_i', 'lambda_F', 'lco_amp', ...
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

            plot_data.w_center = solve_data.w_center;
            plot_data.A_LCO = solve_data.lco_amp;
            plot_data.lambda_F = solve_data.lambda_F;
            plot_data.flutter_idx = solve_data.flutter_onset_idx;
            plot_data.damping_array = solve_data.damping_array;
            plot_data.reduced_freq_array = ...
                FlutterAnalysisCache.nondimentionalize(params, solve_data.natural_frequencies_hz_array);

            vm_surf = max(cat(4, solve_data.vm_upper, solve_data.vm_lower), [], 4);
            vm_pt_time = squeeze(max(vm_surf, [], 3));
            [plot_data.vm_max_p, idx_pt] = max(vm_pt_time, [], 2);

            x_lin = linspace(0, params.a, params.disc_stress);
            y_lin = linspace(-params.b/2, params.b/2, params.disc_stress);
            [Xg, Yg] = meshgrid(x_lin, y_lin);
            x_points = reshape(Xg, 1, []);
            y_points = reshape(Yg, 1, []);

            plot_data.x_max_vm = x_points(idx_pt);
            plot_data.y_max_vm = y_points(idx_pt);
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
