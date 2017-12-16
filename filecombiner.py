# use this file to combine several files

def combineFiles(fpts,combined_filename):
    combined_fpt = open(combined_filename,"w")
    for fpt in fpts:
        content = fpt.read().strip()+"\n"
        combined_fpt.write(content)


def combineWithSuffix(filename_prefixs,suffix):

    actualfilenames = []
    for fpre in filename_prefixs:
        actual_name = fpre +suffix
        actualfilenames.append(actual_name)

    print "combining: "+str(actualfilenames)

    fpts = []
    for filename in actualfilenames:
        newfpt = open(filename,"r")
        fpts.append(newfpt)

    final_file_name = "combined"+suffix
    combineFiles(fpts,final_file_name)




filename_prefixs = ["dev-v1.1","test-any-4","test-add-one-sent","test-add-best-sent"]
combineWithSuffix(filename_prefixs,"-answer-range.txt")
combineWithSuffix(filename_prefixs,"-question.txt")
combineWithSuffix(filename_prefixs,"-story.txt")