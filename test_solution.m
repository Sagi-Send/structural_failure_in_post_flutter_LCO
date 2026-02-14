function [passed_test] = test_solution(tensor_to_test,string)

    loaded_tensor = load('test_solution_data.mat',string);
    
    eps_error = abs(10^-2);     % numerical deviation tolarence
    if tensor_to_test-loaded_tensor.(string) < eps_error
        passed_test = true;
    else
        passed_test = false;
    end

end

