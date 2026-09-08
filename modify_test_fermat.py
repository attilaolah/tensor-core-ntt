with open("tests/test-fermat.cu", "r") as f:
    code = f.read()

# Replace argument parsing
code = code.replace("""    // Parse arguments
    std::string filename = "primes.txt";
    int phase = 2;
    size_t target_digits = 4000;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--file" && i+1 < argc) filename = argv[++i];
        if (arg == "--phase" && i+1 < argc) phase = std::stoi(argv[++i]);
        if (arg == "--digits" && i+1 < argc) target_digits = std::stoi(argv[++i]);
    }""", """    // Parse arguments
    std::string filename = "primes.txt";
    int phase = 2;
    size_t target_bits = 258000;
    int target_index = -1;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--file" && i+1 < argc) filename = argv[++i];
        if (arg == "--phase" && i+1 < argc) phase = std::stoi(argv[++i]);
        if (arg == "--target-bits" && i+1 < argc) target_bits = std::stoull(argv[++i]);
        if (arg == "--index" && i+1 < argc) target_index = std::stoi(argv[++i]);
    }""")

code = code.replace("""        std::string line;
        while (std::getline(primes_file, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (line.length() < 1000) continue; // Skip primes < 1000 decimal digits
            
            mpz_init_set_str(p, line.c_str(), 10);
            
            size_t bit_len = mpz_sizeinbase(p, 2);
            size_t d = (bit_len + 15) / 16;
            
            size_t N_val;
            if (bit_len <= 32000) N_val = 4096;
            else N_val = 65536;
            
            uint64_t inv_n = modulus.invert(N_val);
            std::cout << "Testing prime: " << line.length() << " digits (" << bit_len << " bits) | d=" << d << " | N=" << N_val << "..." << std::endl;
            
            run_fermat_pipeline(p, bit_len, d, N_val, inv_n, modulus, thrust::raw_pointer_cast(precomp_device.get()), constant_precomp);
        }""", """        std::string line;
        std::vector<std::string> candidates;
        while (std::getline(primes_file, line)) {
            if (line.empty() || line[0] == '#') continue;
            candidates.push_back(line);
        }
        
        int selected_idx = -1;
        if (target_index != -1) {
            selected_idx = target_index;
        } else {
            for (size_t i = 0; i < candidates.size(); i++) {
                mpz_init_set_str(p, candidates[i].c_str(), 10);
                size_t bit_len = mpz_sizeinbase(p, 2);
                mpz_clear(p);
                // Allow a small tolerance for target_bits match or take exact
                if (bit_len >= target_bits * 0.95 && bit_len <= target_bits * 1.05) {
                    selected_idx = i;
                    break;
                }
            }
        }
        
        if (selected_idx < 0 || selected_idx >= candidates.size()) {
            std::cout << "Candidate not found!" << std::endl;
            return 1;
        }
        
        line = candidates[selected_idx];
        mpz_init_set_str(p, line.c_str(), 10);
        
        size_t bit_len = mpz_sizeinbase(p, 2);
        size_t exact_digits = line.length();
        size_t d = (bit_len + 15) / 16;
        
        size_t N_val;
        if (bit_len <= 32000) N_val = 4096;
        else N_val = 65536;
        
        if (2*d + 2 > N_val) {
            std::cout << "ERROR: N_val " << N_val << " is not sufficient for 2d+2=" << (2*d+2) << " limbs to prevent aliasing." << std::endl;
            return 1;
        }
        
        uint64_t inv_n = modulus.invert(N_val);
        std::cout << "Testing prime index " << selected_idx << ":" << std::endl;
        std::cout << "exact decimal digit count: " << exact_digits << std::endl;
        std::cout << "bit count: " << bit_len << std::endl;
        std::cout << "d=" << d << " limbs, max convolution length 2d+2=" << (2*d+2) << std::endl;
        std::cout << "Chosen N_val: " << N_val << " (sufficient zero-padding guaranteed)" << std::endl;
        
        run_fermat_pipeline(p, bit_len, d, N_val, inv_n, modulus, thrust::raw_pointer_cast(precomp_device.get()), constant_precomp);
""")

# Progress printing
code = code.replace("""    for (int i = actual_bit_len - 2; i >= 0; --i) {
        cudaGraphLaunch(instance, stream);
        squarings++;
        
        if (mpz_tstbit(p_minus_1, i)) {
            mul_2_kernel<<<blocks, threads, 0, stream>>>(raw_X, raw_T, N_val);
            cudaMemcpyAsync(raw_X, raw_T, N_val * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
            single_block_arbitrary_conditional_sub_p_kernel<<<1, 1024, 0, stream>>>(raw_X, raw_P, d, N_val);
            multiplies++;
        }
    }""", """    int total_steps = actual_bit_len - 1;
    int twenty_percent = total_steps / 5;
    if (twenty_percent == 0) twenty_percent = 1;
    
    for (int i = actual_bit_len - 2; i >= 0; --i) {
        cudaGraphLaunch(instance, stream);
        squarings++;
        
        if (mpz_tstbit(p_minus_1, i)) {
            mul_2_kernel<<<blocks, threads, 0, stream>>>(raw_X, raw_T, N_val);
            cudaMemcpyAsync(raw_X, raw_T, N_val * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
            single_block_arbitrary_conditional_sub_p_kernel<<<1, 1024, 0, stream>>>(raw_X, raw_P, d, N_val);
            multiplies++;
        }
        
        if (squarings % twenty_percent == 0) {
            cudaStreamSynchronize(stream);
            std::cout << "Progress: " << (squarings * 100 / total_steps) << "% (" << squarings << "/" << total_steps << " squarings)" << std::endl;
        }
    }""")

code = code.replace("""    if (mpz_cmp_ui(final_val, 1) == 0) {
        std::cout << "  [PASS] Squarings: " << squarings << " Multiplies: " << multiplies << " | Time: " << ms << " ms (avg " << (ms * 1000.0f / squarings) << " us/step)" << std::endl;
    } else {
        std::cout << "  [FAIL] Final x != 1" << std::endl;
        gmp_printf("         x = %Zd\\n", final_val);
    }""", """    float total_sec = ms / 1000.0f;
    float avg_us = ms * 1000.0f / squarings;
    if (mpz_cmp_ui(final_val, 1) == 0) {
        size_t exact_digits = mpz_sizeinbase(p, 10);
        size_t exact_bits = mpz_sizeinbase(p, 2);
        std::cout << "[PASS] Prime: " << exact_digits << " digits (" << exact_bits << " bits) | Total Time: " << total_sec << " s | Avg per step: " << avg_us << " us" << std::endl;
    } else {
        std::cout << "  [FAIL] Final x != 1" << std::endl;
    }""")

with open("tests/test-fermat.cu", "w") as f:
    f.write(code)
