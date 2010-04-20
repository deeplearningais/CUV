//*LB*
// Copyright (c) 2009, Alexander Krizhevsky
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the University of Toronto 
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*LE*





/*
 * gpu_locking.c
 *
 *  Created on: Dec 2, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "gpu_locking.h"

void _exit_perr(const char* func) {
    perror(func);
    exit(EXIT_FAILURE);
}

void _exit_err(const char* msg) {
    fprintf(stderr, msg);
    exit(EXIT_FAILURE);
}

/*
 * Obtain GPU board lock using Iain's lock-dispensing script.
 */
int get_board_lock() {
    size_t HOST_LEN = 32;
    char* hostName = (char*) malloc(sizeof(char) * HOST_LEN);
    int err;
    while ((err = gethostname(hostName, HOST_LEN)) && HOST_LEN < 1024) {
        HOST_LEN *= 2;
        hostName = (char*) realloc(hostName, HOST_LEN * sizeof(char));
    }
    if (err) {
        _exit_perr("gethostname");
    }
    int boardID = GPU_LOCK_NO_SCRIPT;
    if (!strncmp(hostName, "guppy", HOST_LEN)) {
        int pfdes[2];
        if (pipe(pfdes)) {
            _exit_perr("pipe");
        }
        pid_t pid = fork();
        if (pid == -1) { // err
            _exit_perr("fork");
        } else if (pid == 0) { // child
            close(pfdes[0]); // close read pipe
            dup2(pfdes[1], STDOUT_FILENO);
            close(pfdes[1]);
            execl(GPU_LOCK_SCRIPT, GPU_LOCK_SCRIPT, "--id", NULL);
            _exit_perr("execl");
        }
        // parent
        close(pfdes[1]); // close write pipe
        int BUF_LEN = 2;
        char buf[BUF_LEN];
        int r, totalRead = 0;
        while (totalRead < BUF_LEN && (r = read(pfdes[0], buf + totalRead, BUF_LEN - totalRead)) != 0) {
            if (r == -1) {
                _exit_perr("read");
            } else {
                totalRead += r;
            }
        }
        if (totalRead >= 1) {
            if (sscanf(buf, "%d", &boardID) == 0) {
                _exit_err("sscanf: Unable to read GPU lock script output.\n");
            }
            close(pfdes[0]);
        } else {
            _exit_err("Unable to read GPU lock script output.\n");
        }
    }

    free(hostName);
    return boardID;
}
