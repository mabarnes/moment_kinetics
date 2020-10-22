module initial_conditions

export init_f

using array_allocation: allocate_float
using moment_kinetics_input: initialization_option
using moment_kinetics_input: monomial_degree
using moment_kinetics_input: zwidth, vpawidth

# creates ff and specifies its initial condition
function init_f(z, vpa)
    f = allocate_float(z.n, vpa.n, 3)
    @inbounds begin
        if initialization_option == "gaussian"
            # initial condition is an unshifted Gaussian
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    f[i,j,:] .= (exp(-0.5*((z.grid[i]-3*zwidth)/zwidth)^2)
                     * exp(-0.5*(vpa.grid[j]/vpawidth)^2))
                end
            end
        elseif initialization_option == "monomial"
            # linear variation in z, with offset so that
            # function passes through zero at upwind boundary
            for i ∈ 1:z.n
                f[i,j,:] .= ((z.grid[i] + 0.5*z.L)^monomial_degree
                    .* (vpa.grid[j] + 0.5*vpa.L)^monomial_degree)
            end
        end
        if z.bc == "zero"
            # impose zero incoming BC
            f[1,:,:] .= 0
            #f[nz,:] .= 0
        elseif z.bc == "periodic"
            # impose periodicity
            f[1,:,:] .= f[z.n,:,:]
        end
        if vpa.bc == "zero"
            f[:,1,:] .= 0
        end
    end
    return f
end

end
