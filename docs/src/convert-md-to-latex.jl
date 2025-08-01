"""
Convert a Markdown documentation file to a stripped-down LaTeX file.

Mainly intended to make it easier to edit equations in the docs with a LaTeX editor, so
that they can then be copied back (manually) into the Markdown.
"""
function main()
    if length(ARGS) < 1
        error("Name of .md file to be converted is required as first argument")
    end
    mdname = ARGS[1]
    texname = split(mdname, ".")[1] * ".tex"

    mdtext = readlines(mdname)

    # Start with some preamble to make this a buildable LaTeX document
    textext = "\\documentclass{article}
\\usepackage{amsmath}
\\begin{document}\n"

    ismath = false
    mathtext = ""
    isotherblock = false
    for line ∈ mdtext
        if ismath
            if strip(line) == "```"
                # Skip the closing '```' and append the contents of the `mathtext` block.
                textext *= mathtext
                ismath = false
            else
                # Add the line to `mathtext`
                mathtext *= line * "\n"
            end
        elseif isotherblock
            # Within some 'other' block type, just skip all lines.
            if strip(line) == "```"
                # End of the block, start adding text again.
                isotherblock = false
            end
        elseif length(line) > 1 && all(c == '=' for c ∈ line)
            # Underline indicating section heading. Replace with LaTeX newline.
            textext *= "\\\\\n"
        elseif length(line) > 1 && all(c == '-' for c ∈ line)
            # Underline indicating subsection heading. Replace with LaTeX newline.
            textext *= "\\\\\n"
        elseif strip(line) == "```math"
            # Remove the enclosing '```math' -> '```' from a math block.
            # Here skip the opening '```math' and set `ismath=true` to gather the following
            # lines into `mathtext`.
            ismath = true
            mathtext = ""
        elseif length(strip(line)) ≥ 3 && strip(line)[1:3] == "```"
            # Strip out other kinds of blocks, e.g. raw HTML code
            isotherblock = true
        else
            textext *= line * "\n"
        end
    end

    textext *= "\\end{document}\n"

    open(texname, "w") do io
        print(io, textext)
    end

    return nothing
end

main()
