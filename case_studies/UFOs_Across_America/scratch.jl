# regex to find all version of sec, seconds, secs, etc.

using CSV, DataFramesMeta, Dates
using StatsBase

df = CSV.read("data/ufo_clean.csv", DataFrame)
df.Duration_cleaned = lowercase.(coalesce.(df.Duration, ""))
df.Duration_cleaned = replace.(df.Duration_cleaned, 
  "one"=>"1","two"=>"2","three"=>"3","four"=>"4","five"=>"5","six"=>"6","seven"=>"7","eight"=>"8","nine"=>"9","ten"=>"10",
  "twenty"=>"20","thirty"=>"30","forty"=>"40","fifty"=>"50","sixty"=>"60","seventy"=>"70","eighty"=>"80","ninety"=>"90",
  )

df.Duration_cleaned = let
  v = []

  for i in df.Duration_cleaned

    if occursin("sec", i) && (match(r"\d+", i) != nothing)
      si = match(r"\d+", i).match
      j = Second(parse(Int, si)) 
      push!(v,j)
    
    elseif occursin("min", i) && (match(r"\d+", i) != nothing)
      mi = match(r"\d+", i).match
      j = Minute(parse(Int, mi))
      push!(v, j)

    elseif occursin("hour", i) && (match(r"\d+", i) != nothing)
      hi = match(r"\d+", i).match
      j = Hour(parse(Int, hi))
      push!(v, j)

    elseif occursin("hr", i) && (match(r"\d+", i) != nothing)
      hi = match(r"\d+", i).match
      j = Hour(parse(Int, hi))
      push!(v, j)

    else
      push!(v, i == "" ? missing : i)
    end
  end 
  
  v
end


strs = filter(x -> x != "", [ i for i in df.Duration_cleaned if typeof(i) == String ])
@show strs