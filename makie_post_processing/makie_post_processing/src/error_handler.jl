function makie_post_processing_error_handler(e::Exception, message::String)
    handle_errors = get(input_dict, "handle_errors", true)
    if isa(e, InterruptException) || !handle_errors
        rethrow(e)
    else
        println(message * "\nError was $e.")
        return nothing
    end
end
