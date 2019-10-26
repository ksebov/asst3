/*
* Copyright (c) 1987, 1993, 1994
* The Regents of the University of California.  All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. All advertising materials mentioning features or use of this software
*    must display the following acknowledgement:
*      This product includes software developed by the University of
*      California, Berkeley and its contributors.
* 4. Neither the name of the University nor the names of its contributors
*    may be used to endorse or promote products derived from this software
*    without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
* OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGE.
*/

#include "getopt.h"

#include <stdio.h>
#include <string.h>

/* For communication from `getopt' to the caller.
   When `getopt' finds an option that takes an argument,
   the argument value is returned here. */
const char* optarg = nullptr;

/* Index in ARGV of the next element to be scanned.
   This is used for communication to and from the caller
   and for communication between successive calls to `getopt'.

   On entry to `getopt', zero means this is the first call; initialize.

   When `getopt' returns -1, this is the index of the first of the
   non-option elements that the caller should itself scan.

   Otherwise, `optind' communicates from one call to the next
   how much of ARGV has been scanned so far.  */
int32_t optind = 1;

static bool getopt_initialized = false;
static const char* next_char = nullptr;
/* Callers store zero here to inhibit the error message
   for unrecognized options.  */
int32_t opterr = 1;

/* Set to an option character which was unrecognized.
   This must be initialized on some systems to avoid linking in the
   system's own getopt implementation.  */
int32_t optopt = '?';

/* reset getopt */
static enum { REQUIRE_ORDER, PERMUTE, RETURN_IN_ORDER } ordering;
static int32_t first_nonopt;
static int32_t last_nonopt;

#define SWAP_FLAGS(ch1, ch2)



static void exchange(char** argv) {
  int bottom = first_nonopt;
  int middle = last_nonopt;
  int top = optind;
  char* tem;

  /* Exchange the shorter segment with the far end of the longer segment.
     That puts the shorter segment into the right place.
     It leaves the longer segment in the right place overall,
     but it consists of two parts that need to be swapped next.  */
  while (top > middle && middle > bottom) {
    if (top - middle > middle - bottom) {
      /* Bottom segment is the short one.  */
      int len = middle - bottom;

      /* Swap it with the top part of the top segment.  */
      /*for (register int i = 0; i < len; i++) {*/
      for (int i = 0; i < len; i++) {
        tem = argv[bottom + i];
        argv[bottom + i] = argv[top - (middle - bottom) + i];
        argv[top - (middle - bottom) + i] = tem;
        SWAP_FLAGS(bottom + i, top - (middle - bottom) + i);
      }
      /* Exclude the moved bottom segment from further swapping.  */
      top -= len;
    } else {
      /* Top segment is the short one.  */
      int len = top - middle;

      /* Swap it with the bottom part of the bottom segment.  */
      // for (register int i = 0; i < len; i++) {
      for (int i = 0; i < len; i++) {
        tem = argv[bottom + i];
        argv[bottom + i] = argv[middle + i];
        argv[middle + i] = tem;
        SWAP_FLAGS(bottom + i, middle + i);
      }
      /* Exclude the moved top segment from further swapping.  */
      bottom += len;
    }
  }

  /* Update records for the slots the non-options now occupy.  */

  first_nonopt += (optind - last_nonopt);
  last_nonopt = optind;
}

/* Initialize the internal data when the first call is made.  */
// static const char* getopt_initialize(int argc, char* const* argv, const char* opt_string) {
//  argc = argc;
//  argv = argv;
static const char* getopt_initialize(const char* opt_string) {
  /* Start processing options with ARGV-element 1 (since ARGV-element 0
     is the program name); the sequence of previously skipped
     non-option ARGV-elements is empty.  */

  first_nonopt = last_nonopt = optind;

  next_char = nullptr;

  /* Determine how to handle the ordering of options and nonoptions.  */

  if (opt_string[0] == '-') {
    ordering = RETURN_IN_ORDER;
    ++opt_string;
  } else if (opt_string[0] == '+') {
    ordering = REQUIRE_ORDER;
    ++opt_string;
  } else {
    ordering = PERMUTE;
  }

  return opt_string;
}

/* Scan elements of ARGV (whose length is ARGC) for option characters
   given in opt_string.

   If an element of ARGV starts with '-', and is not exactly "-" or "--",
   then it is an option element.  The characters of this element
   (aside from the initial '-') are option characters.  If `getopt'
   is called repeatedly, it returns successively each of the option characters
   from each of the option elements.

   If `getopt' finds another option character, it returns that character,
   updating `optind' and `next_char' so that the next call to `getopt' can
   resume the scan with the following option character or ARGV-element.

   If there are no more option characters, `getopt' returns -1.
   Then `optind' is the index in ARGV of the first ARGV-element
   that is not an option.  (The ARGV-elements have been permuted
   so that those that are not options now come last.)

   opt_string is a string containing the legitimate option characters.
   If an option character is seen that is not listed in opt_string,
   return '?' after printing an error message.  If you set `opterr' to
   zero, the error message is suppressed but we still return '?'.

   If a char in opt_string is followed by a colon, that means it wants an arg,
   so the following text in the same ARGV-element, or the text of the following
   ARGV-element, is returned in `optarg'.  Two colons mean an option that
   wants an optional arg; if there is text in the current ARGV-element,
   it is returned in `optarg', otherwise `optarg' is set to zero.

   If opt_string starts with `-' or `+', it requests different methods of
   handling the non-option ARGV-elements.
   See the comments about RETURN_IN_ORDER and REQUIRE_ORDER, above.

   Long-named options begin with `--' instead of `-'.
   Their names may be abbreviated as long as the abbreviation is unique
   or is an exact match for some defined option.  If they have an
   argument, it follows the option name in the same ARGV-element, separated
   from the option name by a `=', or else the in next ARGV-element.
   When `getopt' finds a long-named option, it returns 0 if that option's
   `flag' field is nonzero, the value of the option's `val' field
   if the `flag' field is zero.

   The elements of ARGV aren't really const, because we permute them.
   But we pretend they're const in the prototype to be compatible
   with other systems.

   long_opts is a vector of `struct option' terminated by an
   element containing a name which is zero.

   LONGIND returns the index in LONGOPT of the long-named option found.
   It is only valid when a long-named option has been found by the most
   recent call.

   If LONG_ONLY is nonzero, '-' as well as '--' can introduce
   long-named options.  */

static int getopt_internal(int argc, const char* argv[], const char* opt_string,
                           const struct option* long_opts, int32_t* longind, int long_only) {
  int print_errors = opterr;
  if (opt_string[0] == ':') {
    print_errors = 0;
  }

  if (argc < 1) {
    return -1;
  }

  optarg = nullptr;

  if (optind == 0 || !getopt_initialized) {
    if (optind == 0) {
      optind = 1; /* Don't scan ARGV[0], the program name.  */
    }
    opt_string = getopt_initialize(opt_string);
    getopt_initialized = true;
  }

  /* Test whether ARGV[optind] points to a non-option argument.
     Either it does not have option syntax, or there is an environment flag
     from the shell indicating it is not an option.  The later information
     is only used when the used in the GNU libc.  */

#define NONoption_P (argv[optind][0] != '-' || argv[optind][1] == '\0')

  if (next_char == nullptr || *next_char == '\0') {
    /* Advance to the next ARGV-element.  */

    /* Give FIRST_NONOPT & LAST_NONOPT rational values if OPTIND has been
   moved back by the user (who may also have changed the arguments).  */
    if (last_nonopt > optind) {
      last_nonopt = optind;
    }
    if (first_nonopt > optind) {
      first_nonopt = optind;
    }

    if (ordering == PERMUTE) {
      /* If we have just processed some options following some non-options,
         exchange them so that the options come first.  */

      if (first_nonopt != last_nonopt && last_nonopt != optind) {
        exchange(const_cast<char**>(argv));
      } else if (last_nonopt != optind) {
        first_nonopt = optind;
      }

      /* Skip any additional non-options
         and extend the range of non-options previously skipped.  */

      while (optind < argc && NONoption_P) {
        optind++;
      }
      last_nonopt = optind;
    }

    /* The special ARGV-element `--' means premature end of options.
   Skip it like a null option,
   then exchange with previous non-options as if it were an option,
   then skip everything else like a non-option.  */

    if (optind != argc && !strcmp(argv[optind], "--")) {
      optind++;

      if (first_nonopt != last_nonopt && last_nonopt != optind) {
        exchange(const_cast<char**>(argv));
      } else if (first_nonopt == last_nonopt) {
        first_nonopt = optind;
      }
      last_nonopt = argc;

      optind = argc;
    }

    /* If we have done all the ARGV-elements, stop the scan
   and back over any non-options that we skipped and permuted.  */

    if (optind == argc) {
      /* Set the next-arg-index to point at the non-options
         that we previously skipped, so the caller will digest them.  */
      if (first_nonopt != last_nonopt) {
        optind = first_nonopt;
      }
      return -1;
    }

    /* If we have come to a non-option and did not permute it,
   either stop the scan or describe it to the caller and pass it by.  */

    if (NONoption_P) {
      if (ordering == REQUIRE_ORDER) {
        return -1;
      }
      optarg = argv[optind++];
      return 1;
    }

    /* We have found another option-ARGV-element.
   Skip the initial punctuation.  */

    next_char = (argv[optind] + 1 + (long_opts != nullptr && argv[optind][1] == '-'));
  }

  /* Decode the current option-ARGV-element.  */

  /* Check whether the ARGV-element is a long option.

     If long_only and the ARGV-element has the form "-f", where f is
     a valid short option, don't consider it an abbreviated form of
     a long option that starts with f.  Otherwise there would be no
     way to give the -f short option.

     On the other hand, if there's a long option "fubar" and
     the ARGV-element is "-fu", do consider that an abbreviation of
     the long option, just like "--fu", and not "-f" with arg "u".

     This distinction seems to be the most useful approach.  */

  if (long_opts != nullptr &&
      (argv[optind][1] == '-' ||
       (long_only && (argv[optind][2] || !strchr(opt_string, argv[optind][1]))))) {
    const char* name_end;
    const struct option* p;
    const struct option* found_p = nullptr;
    int exact = 0;
    int ambig = 0;
    int found_ind = -1;
    int option_index;

    for (name_end = next_char; *name_end && *name_end != '='; name_end++) {
      /* Do nothing.  */
    }

    /* Test all long options for either exact match
   or abbreviated matches.  */
    for (p = long_opts, option_index = 0; p->name; p++, option_index++) {
      if (!strncmp(p->name, next_char, name_end - next_char)) {
        if ((unsigned int)(name_end - next_char) == (unsigned int)strlen(p->name)) {
          /* Exact match found.  */
          found_p = p;
          found_ind = option_index;
          exact = 1;
          break;
        } else if (found_p == nullptr) {
          /* First nonexact match found.  */
          found_p = p;
          found_ind = option_index;
        } else if (long_only || found_p->has_arg != p->has_arg || found_p->flag != p->flag ||
                   found_p->val != p->val) {
          /* Second or later nonexact match found.  */
          ambig = 1;
        }
      }
    }

    if (ambig && !exact) {
      if (print_errors) {
        fprintf(stderr, ("%s: option `%s' is ambiguous\n"), argv[0], argv[optind]);
      }
      next_char += strlen(next_char);
      optind++;
      optopt = 0;
      return '?';
    }

    if (found_p != nullptr) {
      option_index = found_ind;
      optind++;
      if (*name_end) {
        /* Don't test has_arg with >, because some C compilers don't
       allow it to be used on enums.  */
        if (found_p->has_arg) {
          optarg = name_end + 1;
        } else {
          if (print_errors) {
            if (argv[optind - 1][1] == '-') { /* --option */
              fprintf(stderr, ("%s: option `--%s' doesn't allow an argument\n"), argv[0],
                      found_p->name);
            } else {
              /* +option or -option */
              fprintf(stderr, ("%s: option `%c%s' doesn't allow an argument\n"), argv[0],
                      argv[optind - 1][0], found_p->name);
            }
          }

          next_char += strlen(next_char);

          optopt = found_p->val;
          return '?';
        }
      } else if (found_p->has_arg == 1) {
        if (optind < argc) {
          optarg = argv[optind++];
        } else {
          if (print_errors) {
            fprintf(stderr, ("%s: option `%s' requires an argument\n"), argv[0], argv[optind - 1]);
          }
          next_char += strlen(next_char);
          optopt = found_p->val;
          return opt_string[0] == ':' ? ':' : '?';
        }
      }
      next_char += strlen(next_char);
      if (longind != nullptr) {
        *longind = option_index;
      }

      if (found_p->flag) {
        *(found_p->flag) = found_p->val;
        return 0;
      }
      return found_p->val;
    }

    /* Can't find it as a long option.  If this is not getopt_long_only,
   or the option starts with '--' or is not a valid short
   option, then it's an error.
   Otherwise interpret it as a short option.  */
    if (!long_only || argv[optind][1] == '-' || strchr(opt_string, *next_char) == nullptr) {
      if (print_errors) {
        if (argv[optind][1] == '-') { /* --option */
          fprintf(stderr, ("%s: unrecognized option `--%s'\n"), argv[0], next_char);
        } else {
          /* +option or -option */
          fprintf(stderr, ("%s: unrecognized option `%c%s'\n"), argv[0], argv[optind][0],
            next_char);
        }
      }
      next_char = const_cast<char*>("");
      optind++;
      optopt = 0;
      return '?';
    }
  }

  /* Look at and handle the next short option-character.  */

  {
    char c = *next_char++;
    const char* temp = strchr(opt_string, c);

    /* Increment `optind' when we start to process its last character.  */
    if (*next_char == '\0') {
      ++optind;
    }

    if (temp == nullptr || c == ':') {
      if (print_errors) {
        fprintf(stderr, ("%s: invalid option -- %c\n"), argv[0], c);
      }
      optopt = c;
      return '?';
    }
    /* Convenience. Treat POSIX -W foo same as long option --foo */
    if (temp[0] == 'W' && temp[1] == ';') {
      const char* name_end;
      const struct option* p;
      const struct option* found_p = nullptr;
      int exact = 0;
      int ambig = 0;
      int found_ind = 0;
      int option_index;

      /* This is an option that requires an argument.  */
      if (*next_char != '\0') {
        optarg = next_char;
        /* If we end this ARGV-element by taking the rest as an arg,
           we must advance to the next element now.  */
        optind++;
      } else if (optind == argc) {
        if (print_errors) {
          /* 1003.2 specifies the format of this message.  */
          fprintf(stderr, ("%s: option requires an argument -- %c\n"), argv[0], c);
        }
        optopt = c;
        if (opt_string[0] == ':') {
          c = ':';
        } else {
          c = '?';
        }
        return c;
      } else {
        /* We already incremented `optind' once;
           increment it again when taking next ARGV-elt as argument.  */
        optarg = argv[optind++];
      }

      /* optarg is now the argument, see if it's in the
         table of long_opts.  */

      for (next_char = name_end = optarg; *name_end && *name_end != '='; name_end++) {
        /* Do nothing.  */
      }

      /* Test all long options for either exact match
         or abbreviated matches.  */
      for (p = long_opts, option_index = 0; p->name; p++, option_index++) {
        if (!strncmp(p->name, next_char, name_end - next_char)) {
          if ((unsigned int)(name_end - next_char) == strlen(p->name)) {
            /* Exact match found.  */
            found_p = p;
            found_ind = option_index;
            exact = 1;
            break;
          } else if (found_p == nullptr) {
            /* First nonexact match found.  */
            found_p = p;
            found_ind = option_index;
          } else {
            /* Second or later nonexact match found.  */
            ambig = 1;
          }
        }
      }

      if (ambig && !exact) {
        if (print_errors) {
          fprintf(stderr, ("%s: option `-W %s' is ambiguous\n"), argv[0], argv[optind]);
        }
        next_char += strlen(next_char);
        optind++;
        return '?';
      }
      if (found_p != nullptr) {
        option_index = found_ind;
        if (*name_end) {
          /* Don't test has_arg with >, because some C compilers don't
             allow it to be used on enums.  */
          if (found_p->has_arg) {
            optarg = name_end + 1;
          } else {
            if (print_errors) {
              fprintf(stderr, ("%s: option `-W %s' doesn't allow an argument\n"), argv[0],
                      found_p->name);
            }

            next_char += strlen(next_char);
            return '?';
          }
        } else if (found_p->has_arg == 1) {
          if (optind < argc) {
            optarg = argv[optind++];
          } else {
            if (print_errors) {
              fprintf(stderr, ("%s: option `%s' requires an argument\n"), argv[0],
                      argv[optind - 1]);
            }
            next_char += strlen(next_char);
            return opt_string[0] == ':' ? ':' : '?';
          }
        }
        next_char += strlen(next_char);
        if (longind != nullptr) {
          *longind = option_index;
        }

        if (found_p->flag) {
          *(found_p->flag) = found_p->val;
          return 0;
        }
        return found_p->val;
      }
      next_char = nullptr;
      return 'W'; /* Let the application handle it.   */
    }
    if (temp[1] == ':') {
      if (temp[2] == ':') {
        /* This is an option that accepts an argument optionally.  */
        if (*next_char != '\0') {
          optarg = next_char;
          optind++;
        } else {
          optarg = nullptr;
        }
        next_char = nullptr;
      } else {
        /* This is an option that requires an argument.  */
        if (*next_char != '\0') {
          optarg = next_char;
          /* If we end this ARGV-element by taking the rest as an arg,
             we must advance to the next element now.  */
          optind++;
        } else if (optind == argc) {
          if (print_errors) {
            /* 1003.2 specifies the format of this message.  */
            fprintf(stderr, ("%s: option requires an argument -- %c\n"), argv[0], c);
          }
          optopt = c;
          if (opt_string[0] == ':') {
            c = ':';
          } else {
            c = '?';
          }
        } else {
          /* We already incremented `optind' once;
         increment it again when taking next ARGV-elt as argument.  */
          optarg = argv[optind++];
        }
        next_char = nullptr;
      }
    }
    return c;
  }
}

int getopt(int argc, const char* argv[], const char* opt_string) {
  return getopt_internal(argc, argv, opt_string, nullptr, nullptr, 0);
}

int getopt_long(int argc, const char* argv[], const char* options,
                const struct option* long_options, int32_t* opt_index) {
  return getopt_internal(argc, argv, options, long_options, opt_index, 0);
}

int getopt_long(int argc, char** argv, const char* options,
  const struct option* long_options, int32_t* opt_index) {
  return getopt_internal(argc, (const char**)argv, options, long_options, opt_index, 0);
}

void reset_getopt() {
  optind = 1;
}
