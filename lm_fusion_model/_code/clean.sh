file="data/all_texts.chunked.pyaraby"

# Remove some weird characters
# like <s> and </s>
words_file="data/words_to_remove.txt"

# Use awk to remove words
awk 'NR==FNR { words[$1]; next } 
{
    for (i=1; i<=NF; i++) 
        if (!($i in words)) 
            printf "%s%s", $i, (i==NF ? "\n" : " ")
}' "$words_file" $file.txt > $file.clean.txt