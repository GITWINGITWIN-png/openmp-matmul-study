-- crossref.lua — @fig: / @tab: / @sec: → \ref{...}

local PREFIXES = { fig = true, tab = true, sec = true }

local function is_crossref(id)
  local prefix = id:match("^(%a+):")
  return prefix ~= nil and PREFIXES[prefix] == true
end

function Para(el)
  if #el.content ~= 1 or el.content[1].t ~= "Image" then return nil end
  local img = el.content[1]
  if not img.identifier:match("^fig:") or #img.caption > 0 then return nil end
  local id = img.identifier
  img.identifier = ""
  return pandoc.Figure(pandoc.Plain{img}, {}, pandoc.Attr(id))
end

function Cite(el)
  for _, c in ipairs(el.citations) do
    if not is_crossref(c.id) then return nil end
  end
  local out = {}
  for i, c in ipairs(el.citations) do
    table.insert(out, pandoc.RawInline("latex", "\\ref{" .. c.id .. "}"))
    if i < #el.citations then table.insert(out, pandoc.Str(", ")) end
  end
  return out
end
